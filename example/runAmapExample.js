import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import fetch from 'node-fetch'; // Import node-fetch
import { parseItinerary } from '../src/parser.js';
// import { geocodeAmapAll } from '../src/geocodeAmap.js'; // REMOVED
import { toAmapJson } from '../src/toAmapJson.js';
// import { mcp_amap_maps_streamableHTTP_maps_direction_driving, mcp_amap_maps_streamableHTTP_maps_direction_walking } from '@tool-lib'; // Import direction tools - REMOVED

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const GEOCACHE_PATH = path.resolve(process.cwd(), 'data/geocode-cache.json');
const ROUTE_CACHE_PATH = path.resolve(process.cwd(), 'data/route-cache.json');

async function readJsonCache(filePath) {
    try {
        const content = await fs.readFile(filePath, 'utf-8');
        return JSON.parse(content);
    } catch (error) {
        if (error.code === 'ENOENT') {
            return {}; // File not found, return empty cache
        }
        console.error(`Error reading cache file ${filePath}:`, error);
        return {};
    }
}

async function writeJsonCache(filePath, data) {
    try {
        await fs.writeFile(filePath, JSON.stringify(data, null, 2), 'utf-8');
    } catch (error) {
        console.error(`Error writing cache file ${filePath}:`, error);
    }
}

// --- Start of inlined geocodeAmap.js content ---
// This content was originally in src/geocodeAmap.js

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function geocode(address, key, city = '全国') {
    try {
        const url = `https://restapi.amap.com/v3/geocode/geo?address=${encodeURIComponent(address)}&city=${encodeURIComponent(city)}&key=${key}`;
        const response = await fetch(url);
        const data = await response.json();

        if (data.status === "1" && data.geocodes && data.geocodes.length > 0) {
            const geo = data.geocodes[0];
            return { 
                lng: parseFloat(geo.location.split(',')[0]), 
                lat: parseFloat(geo.location.split(',')[1]),
                address: geo.formatted_address || address,
                province: geo.province,
                city: geo.city,
                district: geo.district,
                adcode: geo.adcode,
            };
        } else {
            console.warn(`[Amap Geocoding] No geocode found for: ${address}, response:`, data);
            return null;
        }
    } catch (error) {
        console.error(`[Amap Geocoding] Failed to geocode ${address}:`, error);
        return null;
    }
}

async function geocodeAmapAll(itinerary, options = {}) {
    const { key, city = '全国', cache = {} } = options;

    if (!key) {
        throw new Error("Amap Web Service Key is required for geocoding.");
    }

    for (const day of itinerary.days) {
        for (const place of day.places) {
            const addressKey = `${place.name}, ${place.address || ''}, ${place.city || day.city || city}`.trim();

            if (cache[addressKey]) {
                console.log(`[Amap Geocoding] Cache hit for: ${addressKey}`);
                place.location = cache[addressKey];
            } else {
                console.log(`[Amap Geocoding] Processing: ${addressKey}`);
                const location = await geocode(`${place.name}, ${place.address}`, key, place.city || day.city || city);
                if (location) {
                    place.location = location;
                    cache[addressKey] = location; // Store in cache
                } else {
                    console.warn(`[Amap Geocoding] Failed to geocode and no cache for: ${addressKey}`);
                }
                await sleep(200); // Gentle API request rate limiting
            }
        }
    }
    return itinerary;
}

// --- End of inlined geocodeAmap.js content ---

async function runAmapExample() {
    const amapWebServiceKey = "9db1a68e5ab0f8b5c8ced8ac11c24411"; // Your Amap Web Service Key
    const amapJsKey = "c81954c7638889562259c2d539b5bbed"; // Your Amap Web JS Key
    const inputPath = './example/input.txt';
    const tripDataPath = './example/output-amap.json';
    const templatePath = path.join(__dirname, 'template.singlefile.html');
    const outputHtmlPath = path.resolve(process.cwd(), 'output-map.html'); // Project root

    if (!amapWebServiceKey || amapWebServiceKey.includes("YOUR_AMAP_WEB_SERVICE_KEY")) {
        console.error("❌ Error: 请在 runAmapExample.js 中填写你的高德 Web 服务 Key");
        process.exit(1);
    }
    if (!amapJsKey || amapJsKey.includes("YOUR_AMAP_WEB_JS_KEY")) {
        console.error("❌ Error: 请在 runAmapExample.js 中填写你的高德 Web JS Key");
        process.exit(1);
    }

    try {
        console.log(`📄 Loading itinerary from ${inputPath}...`);
        const inputText = await fs.readFile(inputPath, 'utf-8');
        const itinerary = parseItinerary(inputText);
        console.log("✅ Itinerary parsed successfully.");

        // Load geocoding cache
        const geocodingCache = await readJsonCache(GEOCACHE_PATH);

        console.log("📍 Geocoding with Amap API (using cache)...");
        const geocodedItinerary = await geocodeAmapAll(itinerary, { key: amapWebServiceKey, city: '全国', cache: geocodingCache });
        console.log("✅ Geocoding complete.");

        // Save updated geocoding cache
        await writeJsonCache(GEOCACHE_PATH, geocodingCache);

        console.log("🧩 Converting to Amap JSON format...");
        const amapJson = toAmapJson(geocodedItinerary, "My Amap Trip");
        console.log("✅ Conversion complete.");

        await fs.writeFile(tripDataPath, JSON.stringify(amapJson, null, 2), 'utf-8');
        console.log(`💾 Amap JSON saved to ${tripDataPath}`);

        // Load route cache
        const routeCache = await readJsonCache(ROUTE_CACHE_PATH);

        console.log("🚗🚶 Pre-caching routes (Driving, Walking)...");
        for (const day of amapJson.days) {
            for (let i = 0; i < day.places.length - 1; i++) {
                const fromPlace = day.places[i];
                const toPlace = day.places[i + 1];

                if (!fromPlace.location || !toPlace.location) continue; // Skip if location is missing

                const fromCoords = `${fromPlace.location.lng},${fromPlace.location.lat}`;
                const toCoords = `${toPlace.location.lng},${toPlace.location.lat}`;

                // Driving route
                const driveCacheKey = `${fromCoords} -> ${toCoords} (drive)`;
                if (!routeCache[driveCacheKey]) {
                    console.log(`  Fetching driving route for: ${fromPlace.name} to ${toPlace.name}`);
                    try {
                        const driveUrl = `https://restapi.amap.com/v3/direction/driving?origin=${fromCoords}&destination=${toCoords}&output=json&key=${amapWebServiceKey}`;
                        const driveResponse = await fetch(driveUrl);
                        const driveData = await driveResponse.json();

                        if (driveData.status === "1" && driveData.route && driveData.route.paths && driveData.route.paths.length > 0) {
                            const path = driveData.route.paths[0].steps.flatMap(step => step.polyline.split(';').map(coords => coords.split(',').map(Number)));
                            routeCache[driveCacheKey] = path;
                        } else {
                            console.warn(`  No driving route found for: ${fromPlace.name} to ${toPlace.name}, response:`, driveData);
                        }
                    } catch (error) {
                        console.error(`  Failed to fetch driving route for ${fromPlace.name} to ${toPlace.name}:`, error);
                    }
                }

                // Walking route
                const walkCacheKey = `${fromCoords} -> ${toCoords} (walk)`;
                if (!routeCache[walkCacheKey]) {
                    console.log(`  Fetching walking route for: ${fromPlace.name} to ${toPlace.name}`);
                    try {
                        const walkUrl = `https://restapi.amap.com/v3/direction/walking?origin=${fromCoords}&destination=${toCoords}&output=json&key=${amapWebServiceKey}`;
                        const walkResponse = await fetch(walkUrl);
                        const walkData = await walkResponse.json();

                        if (walkData.status === "1" && walkData.route && walkData.route.paths && walkData.route.paths.length > 0) {
                            const path = walkData.route.paths[0].steps.flatMap(step => step.polyline.split(';').map(coords => coords.split(',').map(Number)));
                            routeCache[walkCacheKey] = path;
                        } else {
                            console.warn(`  No walking route found for: ${fromPlace.name} to ${toPlace.name}, response:`, walkData);
                        }
                    } catch (error) {
                        console.error(`  Failed to fetch walking route for ${fromPlace.name} to ${toPlace.name}:`, error);
                    }
                }
            }
        }
        console.log("✅ Route pre-caching complete.");

        // Save updated route cache
        await writeJsonCache(ROUTE_CACHE_PATH, routeCache);

        // Read the HTML template
        let htmlTemplate = await fs.readFile(templatePath, 'utf-8');

        // Get trip title (e.g., from the first day or a default)
        const tripTitle = amapJson.title || "我的行程地图";

        // Replace placeholders in the HTML template
        htmlTemplate = htmlTemplate.replace('{{TRIP_TITLE}}', tripTitle);
        htmlTemplate = htmlTemplate.replace('<h3 id="trip-title">{{TRIP_TITLE}}</h3>', `<h3 id="trip-title">${tripTitle}</h3>`);
        htmlTemplate = htmlTemplate.replace('{{TRIP_DATA_JSON}}', JSON.stringify(amapJson, null, 0)); // No pretty print for injection

        // Construct the AMap JS Key injection script tag
        const amapJsKeyInject = `<script src="https://webapi.amap.com/maps?v=2.0&key=${amapJsKey}&plugin=AMap.Driving,AMap.Walking,AMap.Transfer"></script>`;
        htmlTemplate = htmlTemplate.replace('<!-- {{AMAP_JS_KEY_INJECT}} -->', amapJsKeyInject);

        const initScriptCall = String.raw`
            <script>
                const AMAP_WEB_SERVICE_KEY = "${amapWebServiceKey}";
                const AMAP_JS_KEY = "${amapJsKey}";
                window.__routeCache = ${JSON.stringify(routeCache)};
                window.__geocodeCache = ${JSON.stringify(geocodingCache)};

                function ensureDirectRoutes(amapJson) {
                    if (!amapJson || !amapJson.days) return;
                    amapJson.days.forEach(day => {
                        const places = day.places || [];
                        for (let i = 0; i < places.length - 1; i++) {
                            const a = places[i];
                            const b = places[i + 1];
                            if (!a.location || !b.location) continue;
                            const fromCoords = a.location.lng + ',' + a.location.lat;
                            const toCoords = b.location.lng + ',' + b.location.lat;
                            const driveKey = fromCoords + ' -> ' + toCoords + ' (drive)';
                            const walkKey = fromCoords + ' -> ' + toCoords + ' (walk)';
                            const fallbackPath = [[a.location.lng, a.location.lat], [b.location.lng, b.location.lat]];
                            if (!window.__routeCache[driveKey]) {
                                window.__routeCache[driveKey] = { path: fallbackPath, durationMin: null, mode: 'drive' };
                            }
                            if (!window.__routeCache[walkKey]) {
                                window.__routeCache[walkKey] = { path: fallbackPath, durationMin: null, mode: 'walk' };
                            }
                        }
                    });
                }

                function setGenerateStatus(msg, type = '') {
                    const el = document.getElementById('generate-status');
                    if (!el) return;
                    el.className = 'status ' + (type || '');
                    el.textContent = msg || '';
                }

                function parseItineraryText(input) {
                    const lines = (input || '')
                        .split(/\r?\n/)
                        .map(s => s.trim())
                        .filter(Boolean);
                    const dayPatterns = [
                        /^Day\s*(\d+)\s*[:：]?\s*(.*)$/i,
                        /^D\s*(\d+)\s*(.*)$/i,
                        /^第\s*(\d+)\s*天\s*[:：]?\s*(.*)$/
                    ];
                    const placeLineStart = /^[-•·]\s*(.+)$/;
                    const bracketCityRe = /^\[(?<city>[^\]\[]+)\]\s*(?<name>[^()（）]+?)(?:[（(](?<notes>.*?)[)）])?$/;
                    const sepCityRe = /^(?<city>[^·:]+?)\s*[·:]\s*(?<name>[^()（）]+?)(?:[（(](?<notes>.*?)[)）])?$/;

                    const itinerary = { title: '自定义行程', days: [] };
                    let currentDay = null;

                    for (const line of lines) {
                        let matchedDay = false;
                        for (const pattern of dayPatterns) {
                            const match = line.match(pattern);
                            if (match) {
                                const dayNum = parseInt(match[1], 10);
                                const title = (match[2] || '').trim() || ('第' + dayNum + '天');
                                currentDay = { day: dayNum, title, city: null, notes: null, places: [] };
                                itinerary.days.push(currentDay);
                                matchedDay = true;
                                break;
                            }
                        }
                        if (matchedDay) continue;

                        const pm = line.match(placeLineStart);
                        if (pm && currentDay) {
                            const body = pm[1].trim();
                            let city = null, name = null, notes = null;
                            let m = body.match(bracketCityRe);
                            if (m?.groups) {
                                city = (m.groups.city || '').trim();
                                name = (m.groups.name || '').trim();
                                notes = (m.groups.notes || '').trim() || null;
                            } else {
                                m = body.match(sepCityRe);
                                if (m?.groups) {
                                    city = (m.groups.city || '').trim();
                                    name = (m.groups.name || '').trim();
                                    notes = (m.groups.notes || '').trim() || null;
                                } else {
                                    const noteMatch = body.match(/[（(](.*?)[)）]\s*$/);
                                    if (noteMatch) {
                                        notes = noteMatch[1].trim();
                                        name = body.replace(/[（(].*[)）]\s*$/, '').trim();
                                    } else {
                                        name = body;
                                    }
                                }
                            }
                            currentDay.places.push({ name, city: city || null, notes, location: null });
                            if (!currentDay.city && city) currentDay.city = city;
                        }
                    }
                    return itinerary;
                }

                function toAmapJson(itinerary, title = 'My Amap Trip') {
                    return {
                        title,
                        days: (itinerary.days || []).map(day => ({
                            title: day.title || null,
                            city: day.city || null,
                            notes: day.notes ?? null,
                            places: (day.places || []).map(p => ({
                                name: p.name,
                                address: p.address || null,
                                location: p.location || null,
                                type: p.type || null,
                                status: p.status || null,
                                notes: p.notes ?? null,
                                city: p.city ?? null,
                                cityUsedForSearch: p.cityUsedForSearch || null
                            }))
                        }))
                    };
                }

                function inferCityFromTitle(title = '') {
                    const list = ['北京','上海','广州','深圳','杭州','南京','苏州','武汉','西安','成都','重庆','厦门','青岛','天津','哈尔滨','沈阳','长沙','合肥','郑州','福州','三亚','昆明','桂林'];
                    return list.find(c => title.includes(c)) || null;
                }

                async function geocodePlace(name, cityHint) {
                    const cacheKey = name + '|' + (cityHint || '');
                    if (window.__geocodeCache[cacheKey]) return window.__geocodeCache[cacheKey];
                    const queryCity = cityHint || '全国';
                    const url = 'https://restapi.amap.com/v3/geocode/geo?address=' + encodeURIComponent(name) + '&city=' + encodeURIComponent(queryCity) + '&key=' + AMAP_WEB_SERVICE_KEY;
                    const resp = await fetch(url);
                    const data = await resp.json();
                    if (data.status === '1' && data.geocodes && data.geocodes.length > 0) {
                        const geo = data.geocodes[0];
                        const coords = geo.location.split(',');
                        const lng = Number(coords[0]);
                        const lat = Number(coords[1]);
                        const result = {
                            lng,
                            lat,
                            address: geo.formatted_address || name,
                            province: geo.province,
                            city: geo.city || queryCity,
                            district: geo.district,
                            adcode: geo.adcode
                        };
                        window.__geocodeCache[cacheKey] = result;
                        return result;
                    }
                    throw new Error('无法获取 ' + name + ' 的坐标 (status: ' + data.status + ')');
                }

                async function geocodeItinerary(itinerary) {
                    for (const day of itinerary.days) {
                        for (const place of day.places) {
                            if (place.location) continue;
                            try {
                                const cityGuess = place.city || day.city || inferCityFromTitle(day.title) || '';
                                place.location = await geocodePlace(place.name, cityGuess);
                                place.cityUsedForSearch = cityGuess || null;
                                await new Promise(r => setTimeout(r, 180));
                            } catch (err) {
                                console.warn('Geocode failed:', place.name, err);
                            }
                        }
                    }
                    return itinerary;
                }

                function parsePolyline(polylineStr) {
                    if (!polylineStr) return [];
                    return polylineStr.split(';').map(pair => {
                        const [lng, lat] = pair.split(',').map(Number);
                        if (Number.isFinite(lng) && Number.isFinite(lat)) return [lng, lat];
                        return null;
                    }).filter(Boolean);
                }

                async function fetchRouteSegment(from, to, mode = 'drive') {
                    const origin = from.lng + ',' + from.lat;
                    const destination = to.lng + ',' + to.lat;
                    const cacheKey = origin + ' -> ' + destination + ' (' + mode + ')';

                    let url;
                    if (mode === 'drive') {
                        url = 'https://restapi.amap.com/v3/direction/driving?origin=' + origin + '&destination=' + destination + '&output=json&key=' + AMAP_WEB_SERVICE_KEY;
                    } else if (mode === 'walk') {
                        url = 'https://restapi.amap.com/v3/direction/walking?origin=' + origin + '&destination=' + destination + '&output=json&key=' + AMAP_WEB_SERVICE_KEY;
                    } else {
                        return false;
                    }

                    try {
                        const resp = await fetch(url);
                        const data = await resp.json();
                        if (data.status === '1' && data.route && data.route.paths && data.route.paths.length > 0) {
                            const pathData = data.route.paths[0];
                            const steps = pathData.steps || [];
                            const path = steps.flatMap(step => parsePolyline(step.polyline));
                            if (path.length < 2) return false;
                            const seconds = Number(pathData.duration || 0);
                            const minutes = +(seconds / 60 || 0).toFixed(1);
                            window.__routeCache[cacheKey] = { path, durationMin: minutes, mode };
                            return true;
                        }
                    } catch (err) {
                        console.warn('Route fetch failed (' + mode + '):', err);
                    }
                    return false;
                }

                async function updateRoutesForItinerary(amapJson) {
                    if (!amapJson || !amapJson.days) return;
                    for (const day of amapJson.days) {
                        const places = day.places || [];
                        for (let i = 0; i < places.length - 1; i++) {
                            const a = places[i];
                            const b = places[i + 1];
                            if (!a.location || !b.location) continue;
                            await fetchRouteSegment(a.location, b.location, 'drive');
                            await new Promise(r => setTimeout(r, 180));
                            await fetchRouteSegment(a.location, b.location, 'walk');
                            await new Promise(r => setTimeout(r, 180));
                        }
                    }
                }

                async function regenerateMapFromText(rawText) {
                    setGenerateStatus('正在解析行程...', '');
                    const itinerary = parseItineraryText(rawText);
                    if (!itinerary.days.length) {
                        setGenerateStatus('未解析出任何 Day，请检查行程格式。', 'error');
                        return;
                    }
                    setGenerateStatus('正在获取坐标，稍候...', '');
                    await geocodeItinerary(itinerary);
                    const amapJson = toAmapJson(itinerary, itinerary.title || '自定义行程');
                    await updateRoutesForItinerary(amapJson);
                    const tripDataEl = document.getElementById('trip-data');
                    tripDataEl.textContent = JSON.stringify(amapJson);
                    ensureDirectRoutes(amapJson);
                    const updatedRouteCache = await window.AmapTripRenderer.init(amapJson, AMAP_WEB_SERVICE_KEY, AMAP_JS_KEY, window.__routeCache);
                    if (updatedRouteCache) window.__routeCache = updatedRouteCache;
                    setGenerateStatus('✅ 地图已更新', 'success');
                }

                document.addEventListener('DOMContentLoaded', () => {
                    const tripDataEl = document.getElementById('trip-data');
                    const textarea = document.getElementById('itinerary-input');
                    if (textarea && !textarea.value.trim()) {
                        textarea.value = ${JSON.stringify(inputText)};
                    }
                    const btn = document.getElementById('generate-btn');

                    const initialData = JSON.parse(tripDataEl.textContent);
                    (async () => {
                        await updateRoutesForItinerary(initialData);
                        ensureDirectRoutes(initialData);
                        const updatedRouteCache = await window.AmapTripRenderer.init(initialData, AMAP_WEB_SERVICE_KEY, AMAP_JS_KEY, window.__routeCache);
                        if (updatedRouteCache) window.__routeCache = updatedRouteCache;
                    })();

                    if (btn) {
                        btn.addEventListener('click', async () => {
                            const value = (textarea?.value || '').trim();
                            if (!value) {
                                setGenerateStatus('请输入行程文本后再生成地图', 'error');
                                return;
                            }
                            try {
                                const itinerary = parseItineraryText(value);
                                if (!itinerary.days.length) {
                                    setGenerateStatus('未解析出任何 Day，请检查行程格式。', 'error');
                                    return;
                                }
                                await regenerateMapFromText(value);
                            } catch (err) {
                                console.error(err);
                                setGenerateStatus('❌ 生成失败：' + (err.message || err), 'error');
                            }
                        });
                    }
                });
            </script>
        `;
        htmlTemplate = htmlTemplate.replace('<!-- {{INIT_AMAP_TRIP_MAP_CALL}} -->', initScriptCall);

        // Write the final self-contained HTML file
        await fs.writeFile(outputHtmlPath, htmlTemplate, 'utf-8');
        console.log(`✅ Generated map: ${outputHtmlPath}`);

    } catch (error) {
        console.error("❌ An error occurred:", error);
        process.exit(1);
    }
}

runAmapExample();
