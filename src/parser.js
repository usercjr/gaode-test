// src/parser.js
export function parseItinerary(inputText) {
  const days = [];
  const lines = inputText
    .split(/\r?\n/)
    .map(s => s.trim())
    .filter(s => s.length > 0);

  let currentDay = null;

  // 支持：Day 1: 标题 / 第1天 标题 / D1 标题
  const dayPatterns = [
    /^Day\s*(\d+)\s*[:：]?\s*(.*)$/i,
    /^D\s*(\d+)\s*(.*)$/i,
    /^第\s*(\d+)\s*天\s*[:：]?\s*(.*)$/
  ];

  const placeLineStart = /^[-•·]\s*(.+)$/;

  // 支持三种带城市写法（任选其一）：[城市] 地点 / 城市·地点 / 城市:地点
  const bracketCityRe = /^\[(?<city>[^\]\[]+)\]\s*(?<name>[^()（）]+?)(?:[（(](?<notes>.*?)[)）])?$/;
  const sepCityRe = /^(?<city>[^·:]+?)\s*[·:]\s*(?<name>[^()（）]+?)(?:[（(](?<notes>.*?)[)）])?$/;

  for (const line of lines) {
    // Day 行
    let matchedDay = false;
    for (const re of dayPatterns) {
      const m = line.match(re);
      if (m) {
        const dayNum = parseInt(m[1], 10);
        const title = (m[2] || '').trim() || `第${dayNum}天`;
        currentDay = { day: dayNum, title, city: null, notes: null, places: [] };
        days.push(currentDay);
        matchedDay = true;
        break;
      }
    }
    if (matchedDay) continue;

    // 地点行
    const pm = line.match(placeLineStart);
    if (pm && currentDay) {
      const body = pm[1].trim();
      let city = null, name = null, notes = null;

      // 1) [城市] 地点（备注）
      let m = body.match(bracketCityRe);
      if (m?.groups) {
        city = (m.groups.city || '').trim();
        name = (m.groups.name || '').trim();
        notes = (m.groups.notes || '').trim() || null;
      } else {
        // 2) 城市·地点 / 城市:地点（备注）
        m = body.match(sepCityRe);
        if (m?.groups) {
          city = (m.groups.city || '').trim();
          name = (m.groups.name || '').trim();
          notes = (m.groups.notes || '').trim() || null;
        } else {
          // 3) 无城市，提取备注
          const noteMatch = body.match(/[（(](.*?)[)）]\s*$/);
          if (noteMatch) {
            notes = noteMatch[1].trim();
            name = body.replace(/[（(].*[)）]\s*$/, '').trim();
          } else {
            name = body;
          }
        }
      }

      currentDay.places.push({ name, city: city || null, notes });
      // 如果 day 还没 city，且此处提供了 city，则用作 day.city
      if (!currentDay.city && city) currentDay.city = city;
    }
  }

  return { days };
}
