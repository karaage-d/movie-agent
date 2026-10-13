/* ============================================================
   影探 · 电影智能体 — 前端逻辑
   Vue 3（全局构建，Composition API），无需构建工具
   ============================================================ */
const { createApp, ref, computed, onMounted, nextTick, watch } = Vue;

/* ---------- 轻量 Markdown 渲染（先转义，防 XSS）---------- */
function renderMd(src) {
  if (!src) return '';
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));

  // 1) 先抽出围栏代码块（内容只转义，不做行内格式化）
  const blocks = [];
  let text = String(src).replace(/```[a-zA-Z]*\n?([\s\S]*?)```/g, (m, code) => {
    blocks.push(code.replace(/^\n+|\n+$/g, ''));
    return `\u0001${blocks.length - 1}\u0001`;
  });

  // 2) 转义
  text = esc(text);

  // 2) 保护 markdown 链接，避免下一步自动链接重复处理
  const links = [];
  text = text.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, (m, label, url) => {
    links.push(`<a href="${url}" target="_blank" rel="noopener">${label}</a>`);
    return `\u0000${links.length - 1}\u0000`;
  });

  // 3) 裸 URL 自动链接
  text = text.replace(/(https?:\/\/[^\s<>()（）【】]+)/g,
    (m) => `<a href="${m}" target="_blank" rel="noopener">${m}</a>`);

  // 4) 还原 markdown 链接
  text = text.replace(/\u0000(\d+)\u0000/g, (m, i) => links[Number(i)]);

  // 5) 行内样式
  text = text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>');

  // 6) 块级结构
  const lines = text.split('\n');
  const out = [];
  let list = null;
  let para = [];

  const flushPara = () => {
    if (para.length) { out.push(`<p>${para.join('<br>')}</p>`); para = []; }
  };
  const flushList = () => {
    if (list) { out.push(`<${list.tag}>${list.items.join('')}</${list.tag}>`); list = null; }
  };

  for (const raw of lines) {
    const line = raw.trim();
    if (!line) { flushPara(); flushList(); continue; }

    const h = line.match(/^#{1,4}\s+(.*)$/);
    if (h) { flushPara(); flushList(); out.push(`<h3>${h[1]}</h3>`); continue; }

    // 围栏代码块占位
    const fence = line.match(/^\u0001(\d+)\u0001$/);
    if (fence) {
      flushPara(); flushList();
      out.push(`<pre><code>${esc(blocks[Number(fence[1])] || '')}</code></pre>`);
      continue;
    }

    if (/^[-_*—]{3,}$/.test(line)) { flushPara(); flushList(); out.push('<hr>'); continue; }

    const li = line.match(/^\s*[-·•]\s+(.*)$/);
    if (li) {
      flushPara();
      if (!list || list.tag !== 'ul') { flushList(); list = { tag: 'ul', items: [] }; }
      list.items.push(`<li>${li[1]}</li>`);
      continue;
    }

    const oli = line.match(/^\s*\d+[.、)]\s+(.*)$/);
    if (oli) {
      flushPara();
      if (!list || list.tag !== 'ol') { flushList(); list = { tag: 'ol', items: [] }; }
      list.items.push(`<li>${oli[1]}</li>`);
      continue;
    }

    flushList();
    para.push(line);
  }
  flushPara();
  flushList();

  return out.join('');
}

/* ---------- 应用 ---------- */
createApp({
  setup() {
    /* ===== 全局状态 ===== */
    const VIEWS = ['home', 'chat', 'library', 'analysis', 'cinema'];
    const hashView = (location.hash || '').replace(/^#\/?/, '');
    const view = ref(VIEWS.includes(hashView) ? hashView : 'home');
    const navItems = [
      { key: 'home', label: '首页', icon: '🏠' },
      { key: 'chat', label: '对话', icon: '💬' },
      { key: 'library', label: '电影库', icon: '🎞️' },
      { key: 'analysis', label: '数据分析', icon: '📊' },
      { key: 'cinema', label: '附近影院', icon: '🏢' },
    ];

    const status = ref({ ready: false, initializing: true, movies: 0, reviews: 0, model: '', embedding: '', error: null });
    const movies = ref([]);
    const loadingMovies = ref(true);
    const toasts = ref([]);

    /* ===== 接口封装 ===== */
    const api = async (path, opts = {}) => {
      const res = await fetch(path, opts);
      const ct = res.headers.get('content-type') || '';
      if (!ct.includes('json')) throw new Error(`服务返回异常（HTTP ${res.status}）`);
      const data = await res.json();
      if (data && data.error) throw new Error(data.error);
      return data;
    };

    const toast = (text, type = 'ok', ms = 4200) => {
      const id = Date.now() + Math.random();
      toasts.value.push({ id, text, type });
      setTimeout(() => { toasts.value = toasts.value.filter((t) => t.id !== id); }, ms);
    };

    /* ===== 状态轮询 ===== */
    const statusText = computed(() => {
      if (status.value.error && !status.value.ready) return '服务异常';
      if (!status.value.ready) return '初始化中…';
      return `就绪 · ${status.value.movies} 部影片`;
    });
    const statusClass = computed(() => {
      if (status.value.error && !status.value.ready) return 'err';
      if (!status.value.ready) return 'loading';
      return 'ok';
    });

    const pendingAsk = ref('');

    const pollStatus = async () => {
      try {
        const s = await api('/api/status');
        status.value = s;
        if (s.ready) {
          loadMovies();
          // 直接通过 #cinema 深链接进入时，watch 不会触发，这里补一次自动搜索；
          // 放在 ready 之后也避免后端配置尚未加载完就被调用
          if (view.value === 'cinema' && !cinemas.value.length && !cinemaLoading.value) searchCinemas();
          // 支持 ?ask=我要看流浪地球2 深链接：就绪后自动提问
          if (pendingAsk.value) {
            const q = pendingAsk.value;
            pendingAsk.value = '';
            view.value = 'chat';
            draft.value = q;
            nextTick(send);
          }
          return;
        }
        if (s.error) { toast(s.error, 'err', 8000); return; }
        setTimeout(pollStatus, 1200);
      } catch (e) {
        status.value = { ready: false, initializing: false, movies: 0, reviews: 0, error: String(e.message || e) };
        toast('无法连接后端服务，请确认 server/app.py 已启动', 'err', 8000);
      }
    };

    /* ===== 数据加载 ===== */
    const loadMovies = async () => {
      if (movies.value.length) return;
      try {
        const d = await api('/api/movies?limit=500');
        movies.value = d.items || [];
      } catch (e) {
        toast('电影数据加载失败：' + e.message, 'err');
      } finally {
        loadingMovies.value = false;
      }
    };

    /* ===== 统计 ===== */
    const stats = computed(() => {
      const list = movies.value;
      const ratings = list.map((m) => m.douban_rating).filter((v) => v != null);
      const avg = ratings.length ? (ratings.reduce((a, b) => a + b, 0) / ratings.length).toFixed(2) : '—';
      const years = list.map((m) => m.year).filter((v) => v != null);
      const genreSet = new Set();
      list.forEach((m) => String(m.genre || '').split(/[/,，]/).forEach((g) => {
        const t = g.trim();
        if (t && t !== 'nan') genreSet.add(t);
      }));
      return {
        movies: status.value.movies || list.length,
        reviews: status.value.reviews || 0,
        genres: genreSet.size,
        avgRating: avg,
        yearRange: years.length ? `${Math.min(...years)} – ${Math.max(...years)}` : '—',
      };
    });

    const topGenres = computed(() => {
      const count = {};
      movies.value.forEach((m) => String(m.genre || '').split(/[/,，]/).forEach((g) => {
        const t = g.trim();
        if (t && t !== 'nan') count[t] = (count[t] || 0) + 1;
      }));
      return Object.entries(count).sort((a, b) => b[1] - a[1]).slice(0, 11).map(([g]) => g);
    });

    const railMovies = computed(() => movies.value
      .filter((m) => m.poster)
      .sort((a, b) => (b.douban_rating || 0) - (a.douban_rating || 0))
      .slice(0, 16));

    const fmt = (v) => (v == null || Number.isNaN(Number(v)) ? '—' : Number(v).toFixed(1));

    /* ===== 视图切换 ===== */
    const go = (key) => {
      view.value = key;
      if (key === 'library' || key === 'home') loadMovies();
      if (key === 'analysis') loadAnalysis();
    };

    /* ===== 首页 ===== */
    const heroQuery = ref('');
    const hints = ['我要看流浪地球2', '推荐几部高分科幻片', '奥本海默好评率多少', '成都附近有什么电影院'];

    const askFromHero = () => {
      const q = heroQuery.value.trim();
      if (!q) return;
      heroQuery.value = '';
      view.value = 'chat';
      draft.value = q;
      nextTick(send);
    };
    const quickAsk = (q) => {
      view.value = 'chat';
      draft.value = q;
      nextTick(send);
    };

    /* ===== 对话 ===== */
    const messages = ref([]);
    const draft = ref('');
    const thinking = ref(false);
    const chatScroll = ref(null);
    const chatInput = ref(null);

    const chatSubtitle = computed(() => {
      if (status.value.error && !status.value.ready) return '后端未就绪';
      if (!status.value.ready) return '正在加载模型与知识库…';
      const embed = status.value.embedding ? ` · 向量 ${status.value.embedding}` : '';
      return `已连接 ${status.value.model || 'qwen'}${embed} · 知识库 ${status.value.movies} 部影片`;
    });

    const scrollBottom = () => {
      nextTick(() => {
        const el = chatScroll.value;
        if (el) el.scrollTop = el.scrollHeight;
      });
    };

    const autoGrow = (e) => {
      const el = e.target;
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 150) + 'px';
    };

    const send = async () => {
      const text = draft.value.trim();
      if (!text || thinking.value) return;
      messages.value.push({ role: 'user', text });
      draft.value = '';
      if (chatInput.value) chatInput.value.style.height = 'auto';
      thinking.value = true;
      scrollBottom();

      try {
        const d = await api('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text }),
        });
        // MovieAgent.chat() 会把模型异常当作普通回答返回，这里识别出来单独提示
        const answer = String(d.answer || '');
        const isModelError = /模型调用出错|Error code:|Free quota|invalid_api_key|Unauthorized/i.test(answer);
        if (isModelError) {
          const reason = /Free quota exhausted/i.test(answer) ? '通义千问免费额度已用尽（Free quota exhausted）'
            : /invalid_api_key|Unauthorized/i.test(answer) ? 'DashScope API Key 无效或未配置'
            : /arrearage|欠费/i.test(answer) ? '账户欠费'
            : '模型服务暂时不可用';
          messages.value.push({
            role: 'ai',
            text: `⚠️ **模型调用失败**\n\n${reason}\n\n`
              + '- 检查 `config/config.yaml` 里的 `qwen_api_key` 是否有效\n'
              + '- 到 DashScope 控制台确认额度 / 余额后重试\n\n'
              + `**原始报错**\n\n\`\`\`\n${answer}\n\`\`\``,
          });
          toast('模型调用失败：' + reason, 'err', 8000);
        } else {
          // 海报直接跟着对话消息走（原来只存了 trailer，poster 被丢掉了）
          messages.value.push({
            role: 'ai',
            text: answer,
            poster: d.poster || null,
            posterMissing: !!d.poster_missing && !d.poster,
            trailer: d.trailer || null,
          });
          if (d.movie) loadMovieContext(d.movie);
        }      } catch (e) {
        messages.value.push({ role: 'ai', text: `⚠️ ${e.message}` });
      } finally {
        thinking.value = false;
        scrollBottom();
      }
    };

    const resetChat = async () => {
      try { await api('/api/chat/reset', { method: 'POST' }); } catch (e) { /* 忽略 */ }
      messages.value = [];
      currentMovie.value = null;
      movieDetail.value = null;
      toast('对话历史已清空');
    };

    /* ===== 影片上下文 ===== */
    const currentMovie = ref(null);
    const movieDetail = ref(null);
    const posterLoading = ref(false);

    const detailReviews = computed(() => {
      const d = movieDetail.value;
      if (!d || !d.reviews) return [];
      return [...(d.reviews.good || []), ...(d.reviews.bad || [])].slice(0, 5);
    });

    const loadMovieContext = async (title) => {
      posterLoading.value = true;
      try {
        const d = await api('/api/movie?title=' + encodeURIComponent(title));
        currentMovie.value = d.movie;
        movieDetail.value = d;
      } catch (e) { /* 上下文面板失败不打断对话 */ }
      finally { posterLoading.value = false; }
    };

    // 海报文件缺失时 /api/poster 返回 404，避免浏览器留一个裂图
    const onPosterError = () => { if (currentMovie.value) currentMovie.value.poster = null; };

    // 点击对话里的海报缩略图：右侧详情面板拉取大图
    const showPoster = async (title) => {
      if (!title) return;
      await loadMovieContext(title);
      toast(`已载入《${title}》详情`);
    };

    /* ===== 电影库 ===== */
    const libQuery = ref('');
    const libGenre = ref('');
    const libSort = ref('rating');

    const filteredMovies = computed(() => {
      const q = libQuery.value.trim().toLowerCase();
      let list = movies.value;
      if (q) {
        list = list.filter((m) =>
          String(m.title || '').toLowerCase().includes(q) ||
          String(m.director || '').toLowerCase().includes(q) ||
          String(m.actors || '').toLowerCase().includes(q));
      }
      if (libGenre.value) list = list.filter((m) => String(m.genre || '').includes(libGenre.value));

      const sorted = [...list];
      if (libSort.value === 'rating') sorted.sort((a, b) => (b.douban_rating || 0) - (a.douban_rating || 0));
      else if (libSort.value === 'year') sorted.sort((a, b) => (b.year || 0) - (a.year || 0));
      else sorted.sort((a, b) => String(a.title).localeCompare(String(b.title), 'zh'));
      return sorted;
    });

    /* ===== 详情抽屉 ===== */
    const drawerOpen = ref(false);
    const drawerMovie = ref(null);
    const drawerDetail = ref(null);

    const drawerReviews = computed(() => {
      const d = drawerDetail.value;
      if (!d || !d.reviews) return [];
      return [...(d.reviews.good || []), ...(d.reviews.bad || [])].slice(0, 6);
    });

    const openMovie = async (title) => {
      try {
        const d = await api('/api/movie?title=' + encodeURIComponent(title));
        drawerMovie.value = d.movie;
        drawerDetail.value = d;
        drawerOpen.value = true;
      } catch (e) {
        toast('没有找到《' + title + '》的详情', 'err');
      }
    };

    const askAboutMovie = (title) => {
      drawerOpen.value = false;
      view.value = 'chat';
      draft.value = `介绍一下《${title}》的评分和口碑`;
      nextTick(send);
    };

    /* ===== 数据分析 ===== */
    const analysis = ref(null);
    const loadingAnalysis = ref(false);
    const analysisError = ref('');

    const loadAnalysis = async () => {
      if (analysis.value || loadingAnalysis.value) return;
      loadingAnalysis.value = true;
      analysisError.value = '';
      try {
        analysis.value = await api('/api/analysis');
      } catch (e) {
        analysisError.value = e.message;
      } finally {
        loadingAnalysis.value = false;
      }
    };

    /* ===== 附近影院 ===== */
    const city = ref('成都');
    const cinemas = ref([]);
    const cinemaLoading = ref(false);
    const cinemaError = ref('');

    const searchCinemas = async () => {
      const c = city.value.trim();
      if (!c) return;
      cinemaLoading.value = true;
      cinemaError.value = '';
      try {
        const d = await api('/api/cinemas?city=' + encodeURIComponent(c));
        cinemas.value = d.items || [];
        if (!cinemas.value.length) cinemaError.value = `「${c}」周边没找到影院`;
      } catch (e) {
        cinemas.value = [];
        cinemaError.value = e.message;
      } finally {
        cinemaLoading.value = false;
      }
    };

    // 打开外部链接（高德导航 / 购票搜索）。单独抽出来是为了在某些浏览器里
    // 能兜住 window.open 被拦截的情况，同时不动卡片本身的链接。
    const openUrl = (url) => {
      if (!url) return;
      const w = window.open(url, '_blank', 'noopener');
      if (!w) location.href = url;
    };

    /* ===== 生命周期 ===== */
    onMounted(() => {
      const ask = new URLSearchParams(location.search).get('ask');
      if (ask && ask.trim()) pendingAsk.value = ask.trim();
      pollStatus();
      loadAnalysis();
    });

    // 进入影院页时若还没搜索过，自动搜索一次默认城市；并同步 URL 哈希
    watch(view, (v) => {
      if (location.hash !== '#' + v) history.replaceState(null, '', '#' + v);
      if (v === 'cinema' && !cinemas.value.length && !cinemaLoading.value) searchCinemas();
    });

    return {
      view, navItems, go,
      status, statusText, statusClass,
      movies, loadingMovies, stats, topGenres, railMovies, fmt, toasts, toast,
      heroQuery, hints, askFromHero, quickAsk,
      messages, draft, thinking, chatScroll, chatInput, chatSubtitle,
      send, resetChat, autoGrow, scrollBottom,
      currentMovie, movieDetail, detailReviews, loadMovieContext, showPoster,
      libQuery, libGenre, libSort, filteredMovies,
      drawerOpen, drawerMovie, drawerDetail, drawerReviews, openMovie, askAboutMovie,
      analysis, loadingAnalysis, analysisError, loadAnalysis,
      city, cinemas, cinemaLoading, cinemaError, searchCinemas, openUrl,
      renderMd,
    };
  },
}).mount('#app');
