// src/toAmapJson.js
export function toAmapJson(itinerary, title = "My Amap Trip") {
  const out = {
    title,
    days: []
  };

  for (const day of (itinerary.days || [])) {
    out.days.push({
      title: day.title || null,     // ← 保留
      city: day.city || null,       // ← 保留
      notes: day.notes ?? null,
      places: (day.places || []).map(p => ({
        name: p.name,
        address: p.address || null,
        location: p.location || null,  // {lng,lat}
        type: p.type || null,
        status: p.status || null,
        notes: p.notes ?? null,
        city: p.city ?? null,                   // ← place层的城市（如果有）
        cityUsedForSearch: p.cityUsedForSearch || null // ← 实际检索用到的城市
      }))
    });
  }

  return out;
}
