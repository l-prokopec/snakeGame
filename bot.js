const DIRS = [
  { dx: 0, dy: -1, key: 'ArrowUp', name: 'up' },
  { dx: 1, dy: 0, key: 'ArrowRight', name: 'right' },
  { dx: 0, dy: 1, key: 'ArrowDown', name: 'down' },
  { dx: -1, dy: 0, key: 'ArrowLeft', name: 'left' }
];

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const coordKey = (cell) => `${cell.x},${cell.y}`;
const sameCell = (a, b) => !!a && !!b && a.x === b.x && a.y === b.y;
const manhattan = (a, b) => Math.abs(a.x - b.x) + Math.abs(a.y - b.y);

async function waitFor(predicate, { timeout = 5000, interval = 50 } = {}) {
  const start = performance.now();
  while (performance.now() - start < timeout) {
    const value = predicate();
    if (value) return value;
    await sleep(interval);
  }
  return null;
}

class SnakeBot {
  constructor(win) {
    this.win = win;
    this.doc = win.document;
    this.canvas = this.doc.getElementById('game');
    this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
    if (!this.canvas || !this.ctx) {
      throw new Error('Snake canvas not available. Load index.html first.');
    }
    this.grid = 28;
    this.cellSize = this.canvas.width / this.grid;
    this.startCell = { x: Math.floor(this.grid / 2), y: Math.floor(this.grid / 2) };
    this.valueThreshold = 32;
    this.prevSnakeKeys = new Set();
    this.prevHead = null;
    this.expectedHead = null;
    this.commandPending = false;
    this.plan = [];
    this.running = false;
    this.prevSnake = null;
    this.prevFood = null;
    this.lastScore = this.getScore();
    this.currentDir = { x: 1, y: 0 };
    this.lastBodyLength = 1;

    this.startBtn = this.doc.getElementById('startBtn');
    this.againBtn = this.doc.getElementById('againBtn');
    this.restartBtn = this.doc.getElementById('restartBtn');
    this.startOverlay = this.doc.getElementById('startOverlay');
    this.gameOverOverlay = this.doc.getElementById('gameOverOverlay');
  }

  start() {
    if (this.running) return;
    this.running = true;
    console.info('[SnakeBot] starting');
    const step = () => {
      if (!this.running) return;
      try {
        this.update();
      } catch (err) {
        console.error('[SnakeBot] stopped due to error', err);
        this.stop();
        return;
      }
      this.win.requestAnimationFrame(step);
    };
    this.win.requestAnimationFrame(step);
  }

  stop() {
    this.running = false;
  }

  getScore() {
    const raw = this.doc.getElementById('score')?.textContent || '0';
    const val = Number(raw);
    return Number.isFinite(val) ? val : 0;
  }

  expectedLength() {
    return this.getScore() + 1;
  }

  ensureGameRunning() {
    if (this.startOverlay && this.startOverlay.classList.contains('active')) {
      this.startBtn?.click();
      this.prevSnakeKeys.clear();
      this.prevHead = null;
      this.expectedHead = null;
      this.commandPending = false;
      this.plan = [];
      this.prevSnake = null;
      this.prevFood = null;
      this.lastScore = this.getScore();
      this.currentDir = { x: 1, y: 0 };
      this.lastBodyLength = 1;
    }
    if (this.gameOverOverlay && this.gameOverOverlay.classList.contains('active')) {
      this.againBtn?.click();
      this.prevSnakeKeys.clear();
      this.prevHead = null;
      this.expectedHead = null;
      this.commandPending = false;
      this.plan = [];
      this.prevSnake = null;
      this.prevFood = null;
      this.lastScore = this.getScore();
      this.currentDir = { x: 1, y: 0 };
      this.lastBodyLength = 1;
    }
  }

  update() {
    this.ensureGameRunning();
    if (!this.doc.body.classList.contains('playing')) {
      return;
    }

    const currentScore = this.getScore();
    const grewThisTick = currentScore > this.lastScore;
    if (grewThisTick) {
      this.prevFood = null;
    }
    this.lastScore = currentScore;

    let state = this.analyzeBoard();
    if (!state && this.prevSnake && this.prevSnake.length) {
      if (!state && this.expectedHead) {
        const simulated = this.prevSnake.map((seg) => ({ ...seg }));
        simulated.unshift({ x: this.expectedHead.x, y: this.expectedHead.y });
        if (!grewThisTick && simulated.length > 1) {
          simulated.pop();
        }
        this.prevSnake = simulated.map((seg) => ({ ...seg }));
        this.prevSnakeKeys = new Set(this.prevSnake.map(coordKey));
        this.prevHead = this.prevSnake[0];
        this.commandPending = false;
        this.expectedHead = null;
      }
      state = {
        head: this.prevSnake[0],
        tail: this.prevSnake[this.prevSnake.length - 1],
        body: this.prevSnake.slice().map((seg) => ({ ...seg })),
        food: this.prevFood || null
      };
    }

    if (!state) {
      return;
    }

    this.lastBodyLength = state.body.length;
    this.updateCurrentDir(state);

    if ((!this.plan || this.plan.length === 0) && this.shouldAvoidWall(state)) {
      const preferredTurns = this.perpendicularDirs();
      const emergencyTurn = this.chooseSafeDirection(state, preferredTurns);
      if (emergencyTurn) {
        this.plan = [emergencyTurn];
        this.commandPending = false;
      }
    }

    if (this.expectedHead && sameCell(state.head, this.expectedHead)) {
      this.commandPending = false;
      this.prevHead = state.head;
      this.expectedHead = null;
    } else if (!this.prevHead || !sameCell(state.head, this.prevHead)) {
      this.prevHead = state.head;
      this.commandPending = false;
      this.plan = [];
      this.expectedHead = null;
    }

    this.prevSnakeKeys = new Set(state.body.map(coordKey));

    if (!this.plan || this.plan.length === 0) {
      this.plan = this.planPath(state);
    }

    if (!this.plan || this.plan.length === 0) {
      return;
    }

    if (!this.commandPending) {
      const nextDir = this.plan.shift();
      if (nextDir) {
        const dispatched = this.sendDirection(nextDir);
        if (dispatched) {
          this.commandPending = true;
          this.expectedHead = {
            x: state.head.x + nextDir.dx,
            y: state.head.y + nextDir.dy
          };
        }
      }
    }
  }

  sendDirection(dir) {
    if (this.lastBodyLength > 1 && this.currentDir && dir.dx === -this.currentDir.x && dir.dy === -this.currentDir.y) {
      return false;
    }
    const event = new KeyboardEvent('keydown', {
      key: dir.key,
      code: dir.key,
      bubbles: true
    });
    this.win.dispatchEvent(event);
    return true;
  }

  updateCurrentDir(state) {
    if (!state || !state.body || !state.body.length) return;
    if (state.body.length >= 2) {
      const [head, neck] = state.body;
      this.currentDir = { x: head.x - neck.x, y: head.y - neck.y };
      return;
    }
    if (this.expectedHead) {
      const head = state.body[0];
      this.currentDir = { x: this.expectedHead.x - head.x, y: this.expectedHead.y - head.y };
    }
  }

  shouldAvoidWall(state) {
    if (!this.currentDir) return false;
    const next = {
      x: state.head.x + this.currentDir.x,
      y: state.head.y + this.currentDir.y
    };
    return !this.isInside(next);
  }

  perpendicularDirs() {
    if (!this.currentDir) return DIRS;
    if (this.currentDir.x !== 0) {
      return DIRS.filter((dir) => dir.dx === 0);
    }
    if (this.currentDir.y !== 0) {
      return DIRS.filter((dir) => dir.dy === 0);
    }
    return DIRS;
  }

  analyzeBoard() {
    const width = this.canvas.width;
    const height = this.canvas.height;
    if (!width || !height) return null;

    const image = this.ctx.getImageData(0, 0, width, height).data;
    const metrics = [];
    const brightCells = [];
    const expectedLen = Math.max(1, this.expectedLength());

    for (let y = 0; y < this.grid; y++) {
      metrics[y] = [];
      for (let x = 0; x < this.grid; x++) {
        const sample = this.sampleCell(image, width, x, y);
        sample.bright = sample.value > this.valueThreshold && sample.a > 20;
        metrics[y][x] = sample;
        if (sample.bright) {
          brightCells.push({ x, y });
        }
      }
    }

    if (!brightCells.length) {
      return null;
    }

    const clusters = this.buildClusters(metrics);
    if (!clusters.length) return null;

    const snakeCluster = this.identifySnakeCluster(clusters);
    if (!snakeCluster) return null;

    const refinedCluster = this.refineSnakeCluster(snakeCluster, metrics, expectedLen);
    const snake = this.reconstructSnake(refinedCluster, metrics, expectedLen);
    if (!snake) return null;

    const remaining = clusters.filter((c) => c !== snakeCluster);
    const foodCluster = remaining.reduce((best, cluster) => {
      if (!best) return cluster;
      const bestScore = this.clusterBrightness(best, metrics);
      const clusterScore = this.clusterBrightness(cluster, metrics);
      if (Math.abs(clusterScore - bestScore) > 1e-3) {
        return clusterScore > bestScore ? cluster : best;
      }
      return cluster.cells.length < best.cells.length ? cluster : best;
    }, null);
    const food = this.pickFoodCell(foodCluster, metrics);

    if (snake?.body?.length) {
      this.prevSnake = snake.body.map((segment) => ({ ...segment }));
    }
    if (food) {
      this.prevFood = { x: food.x, y: food.y };
    }

    return {
      head: snake.head,
      tail: snake.tail,
      body: snake.body,
      food
    };
  }

  sampleCell(image, width, cellX, cellY) {
    const offsets = [
      [0.35, 0.35],
      [0.65, 0.35],
      [0.35, 0.65],
      [0.65, 0.65]
    ];
    let r = 0;
    let g = 0;
    let b = 0;
    let a = 0;
    for (const [ox, oy] of offsets) {
      const px = Math.min(width - 1, Math.max(0, Math.round((cellX + ox) * this.cellSize)));
      const py = Math.min(this.canvas.height - 1, Math.max(0, Math.round((cellY + oy) * this.cellSize)));
      const idx = (py * width + px) * 4;
      r += image[idx];
      g += image[idx + 1];
      b += image[idx + 2];
      a += image[idx + 3];
    }
    const samples = offsets.length;
    r /= samples;
    g /= samples;
    b /= samples;
    a /= samples;
    const value = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    return { r, g, b, a, value };
  }

  buildClusters(metrics) {
    const visited = new Set();
    const clusters = [];

    const visit = (x, y) => {
      const stack = [[x, y]];
      const cells = [];
      const set = new Set();
      visited.add(coordKey({ x, y }));
      while (stack.length) {
        const [cx, cy] = stack.pop();
        cells.push({ x: cx, y: cy });
        set.add(coordKey({ x: cx, y: cy }));
        const neighbors = [
          [cx + 1, cy],
          [cx - 1, cy],
          [cx, cy + 1],
          [cx, cy - 1]
        ];
        for (const [nx, ny] of neighbors) {
          if (nx < 0 || ny < 0 || nx >= this.grid || ny >= this.grid) continue;
          if (!metrics[ny][nx].bright) continue;
          const key = coordKey({ x: nx, y: ny });
          if (visited.has(key)) continue;
          visited.add(key);
          stack.push([nx, ny]);
        }
      }
      return { cells, set };
    };

    for (let y = 0; y < this.grid; y++) {
      for (let x = 0; x < this.grid; x++) {
        if (!metrics[y][x].bright) continue;
        const key = coordKey({ x, y });
        if (visited.has(key)) continue;
        clusters.push(visit(x, y));
      }
    }

    return clusters;
  }

  identifySnakeCluster(clusters) {
    if (!clusters.length) return null;
    if (this.prevSnakeKeys.size) {
      let best = null;
      let score = -1;
      for (const cluster of clusters) {
        let overlap = 0;
        for (const cell of cluster.cells) {
          if (this.prevSnakeKeys.has(coordKey(cell))) overlap += 1;
        }
        if (overlap > score) {
          score = overlap;
          best = cluster;
        }
      }
      if (best) return best;
    }
    const byCenter = clusters.find((cluster) =>
      cluster.cells.some((c) => sameCell(c, this.startCell))
    );
    if (byCenter) return byCenter;
    return clusters.reduce((largest, cluster) => {
      if (!largest || cluster.cells.length > largest.cells.length) return cluster;
      return largest;
    }, null);
  }

  pickFoodCell(cluster, metrics) {
    if (!cluster) return null;
    return cluster.cells.reduce((best, cell) => {
      const value = metrics[cell.y][cell.x].value;
      if (!best || value > best.value) {
        return { x: cell.x, y: cell.y, value };
      }
      return best;
    }, null);
  }

  clusterBrightness(cluster, metrics) {
    if (!cluster || !cluster.cells.length) return 0;
    let total = 0;
    for (const cell of cluster.cells) {
      total += metrics[cell.y][cell.x].value;
    }
    return total / cluster.cells.length;
  }

  refineSnakeCluster(cluster, metrics, expectedLen) {
    if (!cluster) return null;
    const enriched = cluster.cells.map((cell) => ({
      x: cell.x,
      y: cell.y,
      value: metrics[cell.y][cell.x].value
    }));
    if (!enriched.length) return cluster;
    enriched.sort((a, b) => b.value - a.value);

    const maxKeep = Math.max(expectedLen + 6, Math.min(enriched.length, expectedLen * 2));
    const slice = enriched.slice(0, maxKeep);
    const cutoffIndex = Math.min(expectedLen - 1, slice.length - 1);
    const dynamicThreshold = Math.max(this.valueThreshold, slice[cutoffIndex]?.value ?? this.valueThreshold);
    const filtered = [];
    for (let i = 0; i < slice.length; i++) {
      const cell = slice[i];
      if (cell.value >= dynamicThreshold || i < expectedLen) {
        filtered.push(cell);
      }
    }
    const unique = [];
    const seen = new Set();
    for (const cell of filtered) {
      const key = coordKey(cell);
      if (seen.has(key)) continue;
      seen.add(key);
      unique.push({ x: cell.x, y: cell.y });
      if (unique.length >= maxKeep) break;
    }
    return {
      cells: unique,
      set: new Set(unique.map((c) => coordKey(c)))
    };
  }

  getNeighbors(cell, set) {
    const neighbors = [];
    const options = [
      { x: cell.x + 1, y: cell.y },
      { x: cell.x - 1, y: cell.y },
      { x: cell.x, y: cell.y + 1 },
      { x: cell.x, y: cell.y - 1 }
    ];
    for (const pos of options) {
      if (pos.x < 0 || pos.y < 0 || pos.x >= this.grid || pos.y >= this.grid) continue;
      if (set.has(coordKey(pos))) neighbors.push(pos);
    }
    return neighbors;
  }

  reconstructSnake(cluster, metrics, expectedLen) {
    if (!cluster) return null;
    const set = cluster.set;
    const cells = cluster.cells.map((c) => ({ x: c.x, y: c.y }));
    if (!cells.length) return null;

    let head = null;
    if (this.expectedHead && set.has(coordKey(this.expectedHead))) {
      head = { ...this.expectedHead };
    }
    if (!head && this.prevHead) {
      const candidate = cells.find((c) => manhattan(c, this.prevHead) === 1 && !sameCell(c, this.prevHead));
      if (candidate) head = { ...candidate };
    }
    const neighborCounts = new Map();
    for (const cell of cells) {
      const count = this.getNeighbors(cell, set).length;
      neighborCounts.set(coordKey(cell), count);
    }
    const endpoints = cells.filter((cell) => neighborCounts.get(coordKey(cell)) <= 1);

    if (!head) {
      if (endpoints.length === 1) {
        head = { ...endpoints[0] };
      } else if (endpoints.length > 1) {
        const reference = this.prevHead || this.startCell;
        endpoints.sort((a, b) => manhattan(a, reference) - manhattan(b, reference));
        head = { ...endpoints[0] };
      } else {
        const reference = this.prevHead || this.startCell;
        cells.sort((a, b) => manhattan(a, reference) - manhattan(b, reference));
        head = { ...cells[0] };
      }
    }

    const visited = new Set();
    const body = [];
    let current = head ? { ...head } : { ...cells[0] };
    let prev = null;
    while (current) {
      body.push({ ...current });
      visited.add(coordKey(current));
      const neighbors = this.getNeighbors(current, set);
      let next = null;
      for (const candidate of neighbors) {
        const key = coordKey(candidate);
        if (prev && sameCell(candidate, prev)) continue;
        if (!visited.has(key)) {
          next = candidate;
          break;
        }
      }
      prev = current;
      current = next;
      if (!current) break;
      if (body.length > set.size + 2) {
        return null; // guard against loops when detection fails
      }
    }

    if (body.length > expectedLen) {
      body.length = expectedLen;
    }

    if (body.length < expectedLen && this.prevSnake && this.prevSnake.length) {
      const missing = expectedLen - body.length;
      const supplement = this.prevSnake.slice(-missing);
      for (let i = supplement.length - 1; i >= 0; i--) {
        body.push({ ...supplement[i] });
      }
    }

    if (body.length < expectedLen) {
      return null;
    }

    const tail = body.length ? body[body.length - 1] : head;

    return {
      head: body[0],
      tail,
      body
    };
  }

  planPath(state) {
    if (!state) return [];
    const { body, food } = state;
    if (food) {
      const pathToFood = this.findPath(body, food, 'food');
      if (pathToFood && pathToFood.length) return pathToFood;
    }
    const tail = body[body.length - 1];
    const fallback = this.findPath(body, tail, 'tail');
    if (fallback && fallback.length) return fallback;
    const safe = this.chooseSafeDirection(state);
    return safe ? [safe] : [];
  }

  findPath(body, target, mode) {
    if (!target) return null;
    const maxNodes = 20000;
    const states = [];
    const visited = new Set();

    const startBody = body.map((p) => ({ x: p.x, y: p.y }));
    const startKey = this.bodyKey(startBody);
    states.push({ head: { ...startBody[0] }, body: startBody, parent: -1, move: null });
    visited.add(startKey);

    let cursor = 0;
    while (cursor < states.length && cursor < maxNodes) {
      const state = states[cursor];
      if (sameCell(state.head, target)) {
        if (mode === 'food' || state.parent !== -1) {
          return this.reconstructPath(states, cursor);
        }
      }

      for (const dir of DIRS) {
        const newHead = { x: state.head.x + dir.dx, y: state.head.y + dir.dy };
        if (!this.isInside(newHead)) continue;
        const grows = mode === 'food' && sameCell(newHead, target);
        const collisionBody = grows ? state.body : state.body.slice(0, state.body.length - 1);
        if (collisionBody.some((segment) => segment.x === newHead.x && segment.y === newHead.y)) continue;
        const newBody = [newHead, ...state.body.map((segment) => ({ ...segment }))];
        if (!grows) newBody.pop();
        const key = this.bodyKey(newBody);
        if (visited.has(key)) continue;
        visited.add(key);
        if (states.length < maxNodes) {
          states.push({ head: newHead, body: newBody, parent: cursor, move: dir });
        }
      }

      cursor += 1;
    }

    return null;
  }

  reconstructPath(states, index) {
    const path = [];
    let cursor = index;
    while (cursor !== -1) {
      const state = states[cursor];
      if (state.move) path.push(state.move);
      cursor = state.parent;
    }
    path.reverse();
    return path;
  }

  bodyKey(body) {
    return body.map((segment) => `${segment.x}:${segment.y}`).join('|');
  }

  isInside(pos) {
    return pos.x >= 0 && pos.y >= 0 && pos.x < this.grid && pos.y < this.grid;
  }

  chooseSafeDirection(state, preferred=[]) {
    const seen = new Set();
    const order = [];
    for (const dir of preferred) {
      if (!dir) continue;
      const key = `${dir.dx},${dir.dy}`;
      if (seen.has(key)) continue;
      seen.add(key);
      order.push(dir);
    }
    for (const dir of DIRS) {
      const key = `${dir.dx},${dir.dy}`;
      if (seen.has(key)) continue;
      seen.add(key);
      order.push(dir);
    }

    for (const dir of order) {
      if (this.lastBodyLength > 1 && this.currentDir && dir.dx === -this.currentDir.x && dir.dy === -this.currentDir.y) {
        continue;
      }
      const nx = state.head.x + dir.dx;
      const ny = state.head.y + dir.dy;
      if (!this.isInside({ x: nx, y: ny })) continue;
      const collision = state.body.slice(0, state.body.length - 1).some((seg) => seg.x === nx && seg.y === ny);
      if (!collision) return dir;
    }
    return null;
  }
}

async function bootstrap() {
  if (window.__snakeBotInstance) {
    console.warn('[SnakeBot] instance already running');
    return window.__snakeBotInstance;
  }
  const canvas = await waitFor(() => document.getElementById('game'));
  if (!canvas) {
    console.error('[SnakeBot] canvas not found. Open index.html first.');
    return null;
  }
  const bot = new SnakeBot(window);
  window.__snakeBotInstance = bot;
  bot.start();
  return bot;
}

bootstrap();

export { SnakeBot };
