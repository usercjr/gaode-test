import { DrivingStrategy } from './strategies/driving.js';
import { WalkingStrategy } from './strategies/walking.js';
import { TransitStrategy } from './strategies/transit.js';
import { BaseStrategy } from './strategies/base.js';

export class StrategyRegistry {
  constructor(map) {
    this.map = map;
    this.strategies = {
      drive: new DrivingStrategy(map),
      walk: new WalkingStrategy(map),
      transit: new TransitStrategy(map, null) // city set per-day
    };
    this.currentKey = 'drive';
  }
  set(key) {
    if (!this.strategies[key]) return;
    // clear all strategies overlays when switching
    Object.values(this.strategies).forEach(s => s.clear());
    this.currentKey = key;
  }
  current() { return this.strategies[this.currentKey]; }
}

export async function renderItinerary(map, itinerary, modeKey='drive') {
  const registry = new StrategyRegistry(map);
  registry.set(modeKey);

  // Draw numbered markers and routes day by day
  itinerary.days.forEach((day, dayIdx) => {
    const mode = registry.currentKey;
    const strat = registry.current();

    // if transit, set city for that day
    if (mode === 'transit' && 'setCity' in strat) {
      const city = day.city || inferCityFromTitle(day.title) || '上海';
      strat.setCity(city);
    }

    // add markers
    day.places.forEach((p, idx) => {
      if (p.location) strat.marker(p.location, `${idx+1}`);
    });

    // sequential planning: 1->2, 2->3...
    (async () => {
      for (let i=0; i<day.places.length-1; i++) {
        const a = day.places[i], b = day.places[i+1];
        if (a.location && b.location) {
          try { await strat.plan(a.location, b.location); } catch(e) {}
        }
      }
    })();
  });

  // fit view to all overlays/markers
  setTimeout(() => map.setFitView(null, false, [140, 40, 340, 20]), 200);
}

function inferCityFromTitle(title='') {
  const list = ['北京','上海','广州','深圳','杭州','南京','苏州','武汉','西安','成都','重庆','厦门','青岛','天津','哈尔滨','沈阳','长沙','合肥','郑州','福州','三亚','昆明','桂林'];
  return list.find(c => title.includes(c)) || null;
}
