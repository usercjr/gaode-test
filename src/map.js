(function (global) {
  const DEFAULT_CITY = "北京";
  // AMAP_WEB_SERVICE_KEY is not directly used in map.js as geocoding/routing is pre-processed by runAmapExample.js
  // AMAP_JS_KEY is injected into the HTML script tag.

  // Global map instance and overlays
  let map = null;
  let currentMarkers = [];
  let currentPolylines = [];
  let infoWindow = null;
  let currentDayIndex = 0;
  let currentTravelMode = "drive"; // Default mode

  // AMap services - these will be initialized once the map is ready
  let drivingService = null;
  let walkingService = null;
  let transitService = null;

  // Local route cache, initialized from the data passed by runAmapExample.js
  let routeCache = {};
  const ROUTE_COLORS = ['#1677ff', '#f97316', '#10b981', '#ef4444', '#8b5cf6', '#14b8a6', '#facc15', '#6366f1'];

  // map.js will assume data is already geocoded and routes are generated if not handled by AMap services.

  function normalizeLocation(loc) {
    if (!loc) return null;
    if (typeof loc === "object" && loc.lng != null && loc.lat != null) {
      const lng = Number(loc.lng),
        lat = Number(loc.lat);
      return (Number.isFinite(lng) && Number.isFinite(lat)) ? { lng, lat } : null; // Return object {lng, lat}
    }
    // If it's a string "lng,lat", convert to object
    if (typeof loc === 'string') {
        const [lng, lat] = loc.split(',').map(Number);
        return (Number.isFinite(lng) && Number.isFinite(lat)) ? { lng, lat } : null;
    }
    return null;
  }

  function inferCityFromTitle(title = '') {
    const list = ['北京', '上海', '广州', '深圳', '杭州', '南京', '苏州', '武汉', '西安', '成都', '重庆', '厦门', '青岛', '天津', '哈尔滨', '沈阳', '长沙', '合肥', '郑州', '福州', '三亚', '昆明', '桂林'];
    return list.find(c => title.includes(c)) || null;
  }

  function initMap(containerId, center) {
    map = new AMap.Map(containerId, {
      zoom: 12,
      center: center || [116.39739, 39.90886], // fallback: 北京天安门
      viewMode: "2D",
    });

    // Initialize AMap services here
    drivingService = new AMap.Driving({
      map: map,
      policy: AMap.DrivingPolicy.LEAST_TIME,
    });
    walkingService = new AMap.Walking({
      map: map
    });
    // Transit service city will be set per-day
    transitService = new AMap.Transfer({
      map: map,
      nightflag: false,
      policy: AMap.TransferPolicy.LEAST_TRANSFER, // 公交最少换乘策略
      city: DEFAULT_CITY, // Initial default city, will be updated
    });

    infoWindow = new AMap.InfoWindow({
      offset: new AMap.Pixel(0, -22)
    });

    return map;
  }

  function fitMapTo(lnglats) {
    if (!lnglats || lnglats.length === 0) return;
    const bounds = new AMap.Bounds(
      [Math.min(...lnglats.map((p) => p.lng)), Math.min(...lnglats.map((p) => p.lat))],
      [Math.max(...lnglats.map((p) => p.lng)), Math.max(...lnglats.map((p) => p.lat))]
    );
    map.setBounds(bounds, false, [40, 40, 40, 40]);
  }

  function clearMap() {
    if (map) {
      map.remove(currentMarkers);
      currentPolylines.forEach(poly => {
        if (poly._labelOverlay) {
          map.remove(poly._labelOverlay);
          poly._labelOverlay = null;
        }
      });
      map.remove(currentPolylines);
      currentMarkers = [];
      currentPolylines = [];
      infoWindow.close();
    }
  }

  function renderMarkers(day) {
    day.places.forEach((p, i) => {
      const ll = normalizeLocation(p.location);
      if (!ll) return;

      const marker = new AMap.Marker({
        position: [ll.lng, ll.lat], // Use {lng, lat} object
        label: {
          content: `<div class="poi-badge">${i + 1}</div>`,
          direction: 'top'
        },
      });
      marker.on("click", () => {
        const html = `
                <div style="min-width:220px">
                  <div style="font-weight:600;margin-bottom:4px;">${p.name || "未命名"}</div>
                  <div style="color:#6b7280;font-size:12px;margin-bottom:6px;">${p.address || ""}</div>
                  <div style="font-size:12px;line-height:1.4;">
                    ${p.notes ? ("📝 " + p.notes) : ""}
                    ${p.type ? ("<br/>🏷️ " + p.type) : ""}
                    ${p.status ? ("<br/>📌 " + p.status) : ""}
                  </div>
                </div>
              `;
        infoWindow.setContent(html);
        infoWindow.open(map, marker.getPosition());
      });
      map.add(marker);
      currentMarkers.push(marker);
    });
  }

function toggleRouteDurationLabel(polyline, map) {
  if (!polyline) return;
  if (polyline._labelOverlay) {
    map.remove(polyline._labelOverlay);
    polyline._labelOverlay = null;
    return;
  }

  if (Array.isArray(currentPolylines)) {
    currentPolylines.forEach(other => {
      if (other && other !== polyline && other._labelOverlay) {
        map.remove(other._labelOverlay);
        other._labelOverlay = null;
      }
    });
  }

  const mode = polyline._mode || 'drive';
  const icon = mode === 'drive' ? '🚗' : mode === 'walk' ? '🚶' : '🚌';
  const minutes = polyline._durationMin || 0;
  const text = `${icon} 预计 ${minutes.toFixed(1)} 分钟`;

  const path = polyline.getPath();
  if (!path || !path.length) return;
  const mid = path[Math.floor(path.length / 2)];

  const label = new AMap.Text({
    text,
    position: mid,
    style: {
      'background-color': 'rgba(255,255,255,0.9)',
      'border-radius': '999px',
      'padding': '4px 10px',
      'font-size': '12px',
      'color': '#111',
      'box-shadow': '0 2px 6px rgba(0,0,0,0.15)',
      'border': '1px solid #e5e7eb'
    },
    zIndex: 300
  });

  map.add(label);
  polyline._labelOverlay = label;
}

  async function getRouteDriving(from, to, color = '#1677ff') {
    const cacheKey = `${from.lng},${from.lat} -> ${to.lng},${to.lat} (drive)`;
    const cached = routeCache[cacheKey];
    if (cached && cached.durationMin != null) {
      const cachedPath = cached.path ? cached.path : cached;
      const minutes = cached.durationMin;
      const polyline = new AMap.Polyline({
        path: Array.isArray(cachedPath) ? cachedPath : [],
        strokeColor: color,
        strokeWeight: 5,
        strokeOpacity: 0.9
      });
      polyline._durationMin = minutes;
      polyline._mode = 'drive';
      polyline.on('click', () => toggleRouteDurationLabel(polyline, map));
      map.add(polyline);
      return polyline;
    }

    return new Promise((resolve, reject) => {
      drivingService.search([from.lng, from.lat], [to.lng, to.lat], (status, result) => {
        if (status === 'complete' && result.routes && result.routes[0]) {
          const route = result.routes[0];
          const path = route.steps.flatMap(s => s.path);
          const minutes = +(route.time / 60).toFixed(1);

          const polyline = new AMap.Polyline({
            path,
            strokeColor: color,
            strokeWeight: 5,
            strokeOpacity: 0.9
          });

          polyline._durationMin = minutes;
          polyline._mode = 'drive';
          polyline.on('click', () => toggleRouteDurationLabel(polyline, map));

          map.add(polyline);
          routeCache[cacheKey] = { path, durationMin: minutes, mode: 'drive' };
          resolve(polyline);
        } else if (cached) {
          console.warn('Driving route failed, fallback to cached path without duration:', result);
          const cachedPath = cached.path ? cached.path : cached;
          const polyline = new AMap.Polyline({
            path: Array.isArray(cachedPath) ? cachedPath : [],
            strokeColor: color,
            strokeWeight: 5,
            strokeOpacity: 0.9
          });
          polyline._durationMin = cached.durationMin != null ? cached.durationMin : 0;
          polyline._mode = 'drive';
          polyline.on('click', () => toggleRouteDurationLabel(polyline, map));
          map.add(polyline);
          resolve(polyline);
        } else {
          console.warn('Driving route failed:', result);
          reject(result);
        }
      });
    });
  }

  async function getRouteWalking(from, to, color = '#1677ff') {
    const cacheKey = `${from.lng},${from.lat} -> ${to.lng},${to.lat} (walk)`;
    const cached = routeCache[cacheKey];
    if (cached && cached.durationMin != null) {
      const cachedPath = cached.path ? cached.path : cached;
      const minutes = cached.durationMin;
      const polyline = new AMap.Polyline({
        path: Array.isArray(cachedPath) ? cachedPath : [],
        strokeColor: color,
        strokeWeight: 4,
        strokeStyle: 'dashed',
        strokeOpacity: 0.9
      });
      polyline._durationMin = minutes;
      polyline._mode = 'walk';
      polyline.on('click', () => toggleRouteDurationLabel(polyline, map));
      map.add(polyline);
      return polyline;
    }

    return new Promise((resolve, reject) => {
      walkingService.search([from.lng, from.lat], [to.lng, to.lat], (status, result) => {
        if (status === 'complete' && result.routes && result.routes[0]) {
          const route = result.routes[0];
          const path = route.steps.flatMap(s => s.path);
          const minutes = +(route.time / 60).toFixed(1);

          const polyline = new AMap.Polyline({
            path,
            strokeColor: color,
            strokeWeight: 4,
            strokeStyle: 'dashed',
            strokeOpacity: 0.9
          });

          polyline._durationMin = minutes;
          polyline._mode = 'walk';
          polyline.on('click', () => toggleRouteDurationLabel(polyline, map));

          map.add(polyline);
          routeCache[cacheKey] = { path, durationMin: minutes, mode: 'walk' };
          resolve(polyline);
        } else if (cached) {
          console.warn('Walking route failed, fallback to cached path without duration:', result);
          const cachedPath = cached.path ? cached.path : cached;
          const polyline = new AMap.Polyline({
            path: Array.isArray(cachedPath) ? cachedPath : [],
            strokeColor: color,
            strokeWeight: 4,
            strokeStyle: 'dashed',
            strokeOpacity: 0.9
          });
          polyline._durationMin = cached.durationMin != null ? cached.durationMin : 0;
          polyline._mode = 'walk';
          polyline.on('click', () => toggleRouteDurationLabel(polyline, map));
          map.add(polyline);
          resolve(polyline);
        } else {
          console.warn('Walking route failed:', result);
          reject(result);
        }
      });
    });
  }

  async function getRouteTransit(from, to, city, color = '#1677ff') {
    const cacheKey = `${from.lng},${from.lat} -> ${to.lng},${to.lat} (transit - ${city || DEFAULT_CITY})`;
    const cached = routeCache[cacheKey];
    if (cached && cached.durationMin != null) {
      const cachedPath = cached.path ? cached.path : cached;
      const minutes = cached.durationMin;
      const pathArray = Array.isArray(cachedPath) ? cachedPath : [];
      const flatPath = pathArray.flat ? pathArray.flat() : [].concat(...pathArray);
      const polyline = new AMap.Polyline({
        path: flatPath,
        strokeColor: color,
        strokeWeight: 5,
        strokeOpacity: 0.9
      });
      polyline._durationMin = minutes;
      polyline._mode = 'transit';
      polyline.on('click', () => toggleRouteDurationLabel(polyline, map));
      map.add(polyline);
      return polyline;
    }

    return new Promise((resolve, reject) => {
      transitService.setCity(city || DEFAULT_CITY);
      transitService.search([from.lng, from.lat], [to.lng, to.lat], (status, result) => {
        if (status === 'complete' && result.plans && result.plans[0]) {
          const plan = result.plans[0];
          const minutes = +(plan.time || 0);
          const segPaths = [];
          plan.segments.forEach(seg => {
            if (seg.transit?.path) segPaths.push(seg.transit.path);
            if (seg.walking?.path) segPaths.push(seg.walking.path);
          });
          const flatPath = segPaths.flat();

          const polyline = new AMap.Polyline({
            path: flatPath,
            strokeColor: color,
            strokeWeight: 5,
            strokeOpacity: 0.9
          });

          polyline._durationMin = minutes;
          polyline._mode = 'transit';
          polyline.on('click', () => toggleRouteDurationLabel(polyline, map));

          map.add(polyline);
          routeCache[cacheKey] = { path: segPaths, durationMin: minutes, mode: 'transit' };
          resolve(polyline);
        } else if (cached) {
          console.warn('Transit route failed, fallback to cached path without duration:', result);
          const cachedPath = cached.path ? cached.path : cached;
          const pathArray = Array.isArray(cachedPath) ? cachedPath : [];
          const flatPath = pathArray.flat ? pathArray.flat() : [].concat(...pathArray);
          const polyline = new AMap.Polyline({
            path: flatPath,
            strokeColor: color,
            strokeWeight: 5,
            strokeOpacity: 0.9
          });
          polyline._durationMin = cached.durationMin != null ? cached.durationMin : 0;
          polyline._mode = 'transit';
          polyline.on('click', () => toggleRouteDurationLabel(polyline, map));
          map.add(polyline);
          resolve(polyline);
        } else {
          console.warn('Transit route failed:', result);
          reject(result);
        }
      });
    });
  }

  async function renderRoutes(day, mode) {
    clearMap(); // Use clearMap as per prompt
    renderMarkers(day);

    const routePromises = [];
    const allPoints = [];

    for (let i = 0; i < day.places.length - 1; i++) {
      const fromPlace = day.places[i];
      const toPlace = day.places[i + 1];

      if (fromPlace.location && toPlace.location) {
        const fromLoc = normalizeLocation(fromPlace.location);
        const toLoc = normalizeLocation(toPlace.location);
        if (!fromLoc || !toLoc) {
            console.warn(`Skipping route for invalid locations between ${fromPlace.name} and ${toPlace.name}`);
            continue;
        }

        allPoints.push(fromLoc); // Use {lng, lat} object
        allPoints.push(toLoc);   // Use {lng, lat} object

        const color = ROUTE_COLORS[i % ROUTE_COLORS.length];

        let routePromise;
        if (mode === "drive") {
          routePromise = getRouteDriving(fromLoc, toLoc, color);
        } else if (mode === "walk") {
          routePromise = getRouteWalking(fromLoc, toLoc, color);
        } else {
          routePromise = getRouteDriving(fromLoc, toLoc, color);
        }
        routePromises.push(routePromise);
      } else {
          console.warn(`Skipping route for missing location between ${fromPlace.name || 'Unknown'} and ${toPlace.name || 'Unknown'}`);
      }
    }

    const routes = await Promise.allSettled(routePromises);
    routes.forEach((result) => {
      if (result.status === "fulfilled" && result.value) {
        map.add(result.value);
        currentPolylines.push(result.value);
      } else if (result.status === "rejected") {
        console.error("Failed to get route:", result.reason); // Log rejection reason
      }
    });

    if (allPoints.length > 0) {
      fitMapTo(allPoints);
    } else if (day.places.length > 0 && day.places[0].location) {
      // If only one marker, center on it
      const firstLoc = normalizeLocation(day.places[0].location);
      if (firstLoc) {
          map.setCenter([firstLoc.lng, firstLoc.lat]);
          map.setZoom(14);
      }
    } else {
        console.warn("No valid locations to fit map or center.");
    }
  }


  function renderDayList(days) {
    const dayListEl = document.getElementById("day-list");
    dayListEl.innerHTML = ""; // Clear existing buttons

    days.forEach((day, index) => {
      const button = document.createElement("button");
      button.className = "day-item" + (index === currentDayIndex ? " active" : ""); // Use day-item class
      button.innerHTML = `<span>${day.title}</span><span class="meta">${day.places.length} 个点</span>`;
      button.onclick = () => {
        currentDayIndex = index;
        // Update active state for day buttons
        document.querySelectorAll("#day-list .day-item").forEach(btn => btn.classList.remove("active"));
        button.classList.add("active");
        renderRoutes(days[currentDayIndex], currentTravelMode);
      };
      dayListEl.appendChild(button);
    });
  }

  function setupModeSwitching(days) {
    const modeButtonsContainer = document.getElementById("mode-buttons");
    modeButtonsContainer.innerHTML = ''; // Clear existing buttons

    const modes = [
        { label: '🚶 步行', mode: 'walk' },
        { label: '🚗 驾车', mode: 'drive' }
    ];

    modes.forEach(m => {
        const button = document.createElement("button");
        button.className = "mode-btn " + (m.mode === currentTravelMode ? "active" : ""); // Use mode-btn class
        button.textContent = m.label;
        button.dataset.mode = m.mode;
        button.onclick = () => {
            document.querySelectorAll("#mode-buttons .mode-btn").forEach(btn => btn.classList.remove("active"));
            button.classList.add("active");
            currentTravelMode = m.mode;
            renderRoutes(days[currentDayIndex], currentTravelMode);
        };
        modeButtonsContainer.appendChild(button);
    });
  }

  const AmapTripRenderer = {
    // This function is called from the HTML after DOMContentLoaded
    init: async function(tripDataJson, amapWebServiceKey, amapJsKey, initialRouteCache = {}) {
      routeCache = initialRouteCache; // Initialize route cache

      // Note: amapWebServiceKey and amapJsKey are passed but not directly used in map.js
      // as data preprocessing and AMap JS API loading are handled externally.
      const tripData = tripDataJson; // Direct use of parsed JSON

      if (!tripData || !tripData.days || tripData.days.length === 0) {
        console.error("未读取到行程数据或行程数据为空。");
        return;
      }

      const days = tripData.days;

      let initialCenter = [116.39739, 39.90886]; // Default to Beijing Tiananmen
      if (days[0] && days[0].places.length > 0 && days[0].places[0].location) {
        const firstLoc = normalizeLocation(days[0].places[0].location);
        if (firstLoc) {
          initialCenter = [firstLoc.lng, firstLoc.lat]; // initMap expects array
        }
      }

      initMap("map", initialCenter); // Use "map" as container ID
      renderDayList(days);
      setupModeSwitching(days);

      // Render the first day's routes with the default mode
      await renderRoutes(days[currentDayIndex], currentTravelMode);
      return routeCache; // Return the potentially updated cache
    },
    initMap,
    renderDayList,
    renderMarkers,
    renderRoutes,
    clearMap,
    setupModeSwitching,
    getRouteCache: function() {
        return routeCache; // Expose a method to get the current state of the route cache
    }
  };

  global.AmapTripRenderer = AmapTripRenderer;

})(window);
