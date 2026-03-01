import fs from 'fs/promises';
import path from 'path';

const CACHE_DIR = './data';

async function ensureCacheDir() {
  try {
    await fs.mkdir(CACHE_DIR, { recursive: true });
  } catch (error) {
    if (error.code !== 'EEXIST') {
      console.error("Error ensuring cache directory exists:", error);
      throw error;
    }
  }
}

async function getCache(cacheType, key) {
  await ensureCacheDir();
  const cacheFilePath = path.join(CACHE_DIR, `cache.${cacheType}.json`);
  try {
    const cacheContent = await fs.readFile(cacheFilePath, 'utf-8');
    const cache = JSON.parse(cacheContent);
    return cache[key];
  } catch (error) {
    if (error.code === 'ENOENT') {
      // Cache file does not exist, return null
      return null;
    }
    console.error(`Error reading ${cacheType} cache for key "${key}":`, error);
    return null;
  }
}

async function setCache(cacheType, key, value) {
  await ensureCacheDir();
  const cacheFilePath = path.join(CACHE_DIR, `cache.${cacheType}.json`);
  let cache = {};
  try {
    const cacheContent = await fs.readFile(cacheFilePath, 'utf-8');
    cache = JSON.parse(cacheContent);
  } catch (error) {
    if (error.code !== 'ENOENT') {
      console.warn(`Warning: Could not read existing ${cacheType} cache, creating new one. Error:`, error);
    }
  }

  cache[key] = value;

  try {
    await fs.writeFile(cacheFilePath, JSON.stringify(cache, null, 2), 'utf-8');
  } catch (error) {
    console.error(`Error writing ${cacheType} cache for key "${key}":`, error);
    throw error;
  }
}

export { getCache, setCache };
