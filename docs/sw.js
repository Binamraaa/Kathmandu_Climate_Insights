/* के लाउने? service worker: works offline with the last forecast it saw.
   Network first for everything (updates show immediately when online),
   falling back to the cached copy when there is no connection. */
const CACHE = "klaune-v1";
const SHELL = ["./", "index.html", "style.css", "app.js", "manifest.webmanifest",
               "icons/icon-192.png", "icons/icon-512.png", "forecast.json"];

self.addEventListener("install", e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).catch(() => {}));
  self.skipWaiting();
});

self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))));
  self.clients.claim();
});

self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin) return;  // fonts etc. go to the network
  e.respondWith(
    fetch(e.request).then(res => {
      if (res.ok) { const copy = res.clone(); caches.open(CACHE).then(c => c.put(e.request, copy)); }
      return res;
    }).catch(() => caches.match(e.request, {ignoreSearch: true}))
  );
});
