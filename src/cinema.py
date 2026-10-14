# -*- coding: utf-8 -*-
"""电影院搜索与购票链接，数据来自高德地图 Web API。"""
import urllib.parse

import requests


def amap_marker_url(name: str, location: str = "") -> str:
    """高德地图标注页。有坐标就精确定位，否则按影院名搜索。"""
    name = str(name or "").strip()
    if location and "," in location:
        return "https://uri.amap.com/marker?position={}&name={}".format(
            urllib.parse.quote(location, safe=","), urllib.parse.quote(name or "影院"))
    return "https://www.amap.com/search?query=" + urllib.parse.quote(name or "电影院")


def amap_navigation_url(name: str, location: str) -> str:
    """高德导航链接，桌面浏览器打开网页版路线规划。"""
    if not location or "," not in location:
        return ""
    return "https://uri.amap.com/navigation?to={},{}&mode=car&coordinate=gaode&callnative=0".format(
        urllib.parse.quote(location, safe=","), urllib.parse.quote(name or "影院"))


def ticket_search_url(cinema_name: str) -> str:
    """购票搜索链接。猫眼对爬虫返回 403，但这是给浏览器点击的跳转链接。"""
    kw = f"{str(cinema_name or '').strip()} 电影票"
    return "https://www.maoyan.com/query?kw=" + urllib.parse.quote(kw)


def search_nearby_cinemas(amap_key: str, city: str, movie_name: str = "",
                          location: str = "", radius: int = 5000,
                          page: int = 1) -> list:
    """
    使用高德地图 API 搜索附近电影院

    参数:
        amap_key: 高德地图 Web API Key
        city: 城市名称（如"成都"）或城市编码（如"028"）
        movie_name: 电影名（可选，用于后续排片查询）
        location: 中心点坐标 "经度,纬度"（可选，不传则用城市中心）
        radius: 搜索半径（米），默认 5000
        page: 页码，默认 1

    返回:
        list[dict]: 电影院列表，每项含 name/address/location/distance
                    以及 map_url（地图页）/nav_url（导航）/ticket_url（购票搜索）
    """
    if not amap_key or amap_key == "YOUR_AMAP_KEY_HERE":
        return [{"error": "请先在 config/config.yaml 中配置 amap_key"}]

    # 1. 如果没给坐标，先通过城市名获取城市中心坐标
    if not location:
        geo_url = "https://restapi.amap.com/v3/geocode/geo"
        geo_params = {
            "key": amap_key,
            "address": city,
            "output": "JSON",
        }
        try:
            geo_resp = requests.get(geo_url, params=geo_params, timeout=10)
            geo_data = geo_resp.json()
            if geo_data.get("status") == "1" and geo_data.get("geocodes"):
                location = geo_data["geocodes"][0].get("location", "")
        except Exception:
            pass

    if not location:
        return [{"error": f"无法定位城市「{city}」，请检查城市名称"}]

    # 2. 周边搜索：电影院（POI 大类 080600）
    search_url = "https://restapi.amap.com/v3/place/around"
    params = {
        "key": amap_key,
        "location": location,
        "keywords": "电影院",
        "types": "080600",
        "radius": radius,
        "offset": 20,
        "page": page,
        "extensions": "all",
        "output": "JSON",
    }

    try:
        resp = requests.get(search_url, params=params, timeout=10)
        data = resp.json()
    except Exception as e:
        return [{"error": f"高德 API 请求失败: {e}"}]

    if data.get("status") != "1":
        return [{"error": f"高德 API 返回错误: {data.get('info', '未知错误')}"}]

    pois = data.get("pois", [])
    if not pois:
        return [{"error": f"在「{city}」周边 {radius} 米内未找到电影院"}]

    cinemas = []
    for p in pois:
        dist = p.get("distance", "未知")
        dist_str = f"{int(dist)}m" if dist and dist != "未知" else "未知"
        name = p.get("name", "未知影院")
        loc = p.get("location", "") or ""
        cinemas.append({
            "name": name,
            "address": p.get("address", "未知地址"),
            "location": loc,
            "distance": dist_str,
            "tel": p.get("tel", "") or "",
            "rating": p.get("biz_ext", {}).get("rating", "") or "",
            "map_url": amap_marker_url(name, loc),
            "nav_url": amap_navigation_url(name, loc),
            "ticket_url": ticket_search_url(name),
        })

    return cinemas

def format_cinemas_for_agent(cinemas: list, city: str) -> str:
    """将电影院列表格式化为 Agent 可读的文本"""
    if not cinemas:
        return f"未找到「{city}」附近的电影院。"
    if "error" in cinemas[0]:
        return cinemas[0]["error"]

    lines = [f"🏢 **{city}** 附近电影院：\n"]
    for i, c in enumerate(cinemas, 1):
        lines.append(
            f"{i}. **{c['name']}** — {c['address']}（距中心{c['distance']}）"
        )
    return "\n".join(lines)