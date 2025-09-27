const DIRS = [
  { dx: 0, dy: -1, key: 'ArrowUp', name: 'up' },
  { dx: 1, dy: 0, key: 'ArrowRight', name: 'right' },
  { dx: 0, dy: 1, key: 'ArrowDown', name: 'down' },
  { dx: -1, dy: 0, key: 'ArrowLeft', name: 'left' }
];

const TURN_ACTIONS = [-1, 0, 1];
const STEP_PENALTY_PLAY = -0.01;
const STEP_PENALTY_TRAIN = -0.02;
const FOOD_REWARD = 1.0;
const DEATH_PENALTY = -1.0;
const DIST_SHAPING = 0.003;
const STALL_PENALTY_DEFAULT = -0.2;
const NO_PROGRESS_LIMIT_DEFAULT = 280;
const EPSILON_DECAY_DEFAULT = 0.995;
const EPSILON_MIN_DEFAULT = 0.02;
const ALPHA_DEFAULT = 0.3;
const ALPHA_TRAIN_DEFAULT = 0.35;
const ALPHA_DECAY_DEFAULT = 0.999;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const coordKey = (cell) => `${cell.x},${cell.y}`;
const sameCell = (a, b) => !!a && !!b && a.x === b.x && a.y === b.y;
const manhattan = (a, b) => Math.abs(a.x - b.x) + Math.abs(a.y - b.y);

class QAgent {
  constructor() {
    this.table = new Map();
    this.alpha = ALPHA_DEFAULT;
    this.gamma = 0.9;
    this.epsilon = 0.25;
    this.load();
  }

  keyOf(state) {
    return [
      state.dFront,
      state.dLeft,
      state.dRight,
      state.dir,
      state.foodX,
      state.foodY
    ].join('|');
  }

  getQ(key) {
    if (!this.table.has(key)) {
      this.table.set(key, new Float32Array([0, 0, 0]));
    }
    return this.table.get(key);
  }

  chooseAction(state) {
    const key = this.keyOf(state);
    const q = this.getQ(key);
    if (Math.random() < this.epsilon) {
      return Math.floor(Math.random() * TURN_ACTIONS.length);
    }
    if (q[1] >= q[0] && q[1] >= q[2]) return 1;
    return q[0] >= q[2] ? 0 : 2;
  }

  update(prevState, actionIndex, reward, nextState, done) {
    if (!prevState) return;
    const key = this.keyOf(prevState);
    const q = this.getQ(key);
    const oldVal = q[actionIndex];

    let target = reward;
    if (!done && nextState) {
      const nextKey = this.keyOf(nextState);
      const nextQ = this.getQ(nextKey);
      const maxNext = Math.max(nextQ[0], nextQ[1], nextQ[2]);
      target += this.gamma * maxNext;
    }
    q[actionIndex] = (1 - this.alpha) * oldVal + this.alpha * target;
  }

  decayEpsilon(min=0.02, factor=0.995) {
    this.epsilon = Math.max(min, this.epsilon * factor);
  }

  save() {
    const obj = {};
    for (const [k, v] of this.table.entries()) obj[k] = Array.from(v);
    try {
      localStorage.setItem('snake_q_table', JSON.stringify(obj));
    } catch (err) {
      console.warn('[QAgent] Failed to save table', err);
    }
  }

  load() {
    try {
      const raw = localStorage.getItem('snake_q_table');
      if (!raw) return;
      const parsed = JSON.parse(raw);
      this.table = new Map(
        Object.entries(parsed).map(([k, arr]) => [k, Float32Array.from(arr)])
      );
    } catch (err) {
      console.warn('[QAgent] Failed to load table', err);
    }
  }
}

class SnakeQLearner {
  constructor(win) {
    this.win = win;
    this.doc = win.document;
    this.canvas = this.doc.getElementById('game');
    this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
    if (!this.canvas || !this.ctx) {
      throw new Error('Snake canvas not available. Load index.html first.');
    }
    this.GRID = 28;
    this.cellSize = this.canvas.width / this.GRID;
    this.agent = new QAgent();
    this.mode = 'idle';
    this.running = false;

    this.prevObservation = null;
    this.prevState = null;
    this.prevActionIndex = null;
    this.prevScore = this.getScore();
    this.resetEpisodeStats();
    this.noProgressLimit = NO_PROGRESS_LIMIT_DEFAULT;
    this.stallPenalty = STALL_PENALTY_DEFAULT;
    this.alphaDecay = ALPHA_DECAY_DEFAULT;
    this.stepPenalty = STEP_PENALTY_PLAY;
    this.episodeScores = [];
    this.scoreWindow = [];
    this.lastWindowAverage = null;

    this.completedEpisodes = 0;
    this.targetEpisodes = 0;
    this.epsilonMin = EPSILON_MIN_DEFAULT;
    this.epsilonDecay = EPSILON_DECAY_DEFAULT;
    this.autoSaveInterval = 100;

    this.loop = this.loop.bind(this);
    this.ensureGameHandlers();
  }

  ensureGameHandlers() {
    const startOverlay = this.doc.getElementById('startOverlay');
    if (startOverlay && startOverlay.classList.contains('active')) {
      this.doc.getElementById('startBtn')?.click();
    }
  }

  getScore() {
    const raw = this.doc.getElementById('score')?.textContent || '0';
    const val = Number(raw);
    return Number.isFinite(val) ? val : 0;
  }

  isPlaying() {
    return this.doc.body.classList.contains('playing');
  }

  isGameOver() {
    return this.doc.getElementById('gameOverOverlay')?.classList.contains('active');
  }

  async resetGame() {
    const restartBtn = this.doc.getElementById('restartBtn');
    const startBtn = this.doc.getElementById('startBtn');
    const againBtn = this.doc.getElementById('againBtn');

    if (this.isGameOver()) {
      againBtn?.click();
      await sleep(80);
    } else if (!this.isPlaying()) {
      startBtn?.click();
      await sleep(80);
    } else {
      restartBtn?.click();
      await sleep(80);
    }
    this.prevObservation = null;
    this.prevState = null;
    this.prevActionIndex = null;
    this.prevScore = this.getScore();
    this.resetEpisodeStats();
  }

  startLoop() {
    if (this.running) return;
    this.running = true;
    this.win.requestAnimationFrame(this.loop);
  }

  stopLoop() {
    this.running = false;
  }

  resetEpisodeStats() {
    this.prevDistance = null;
    this.lastHeadKey = null;
    this.currentDir = { dx: 1, dy: 0 };
    this.noProgressSteps = 0;
    this.bestDistance = Infinity;
    this.currentEpisodeScore = 0;
  }

  async train({
    episodes = 50,
    epsilon = 0.3,
    epsilonMin = EPSILON_MIN_DEFAULT,
    epsilonDecay = EPSILON_DECAY_DEFAULT,
    alpha = ALPHA_TRAIN_DEFAULT,
    alphaDecay = ALPHA_DECAY_DEFAULT,
    stepPenalty = STEP_PENALTY_TRAIN,
    noProgressLimit = NO_PROGRESS_LIMIT_DEFAULT,
    stallPenalty = STALL_PENALTY_DEFAULT
  } = {}) {
    this.mode = 'train';
    this.targetEpisodes = episodes;
    this.completedEpisodes = 0;
    this.agent.epsilon = epsilon;
    this.epsilonMin = epsilonMin;
    this.epsilonDecay = epsilonDecay;
    this.agent.alpha = alpha;
    this.alphaDecay = alphaDecay;
    this.stepPenalty = stepPenalty;
    this.noProgressLimit = noProgressLimit;
    this.stallPenalty = stallPenalty;
    this.resetEpisodeStats();
    this.episodeScores = [];
    this.scoreWindow = [];
    this.lastWindowAverage = null;
    await this.resetGame();
    this.startLoop();
    return new Promise((resolve) => {
      const check = () => {
        if (this.completedEpisodes >= this.targetEpisodes) {
          resolve();
        } else {
          this.win.requestAnimationFrame(check);
        }
      };
      this.win.requestAnimationFrame(check);
    });
  }

  async play({ epsilon = 0.02 } = {}) {
    this.mode = 'play';
    this.agent.epsilon = epsilon;
    this.stepPenalty = STEP_PENALTY_PLAY;
    this.noProgressLimit = Infinity;
    this.stallPenalty = 0;
    this.alphaDecay = 1;
    await this.resetGame();
    this.startLoop();
  }

  stop() {
    this.mode = 'idle';
    this.stepPenalty = STEP_PENALTY_PLAY;
    this.stopLoop();
  }

  loop() {
    if (!this.running) return;

    try {
      this.update();
    } catch (err) {
      console.error('[SnakeQLearner] stopped due to error', err);
      this.stop();
      return;
    }

    if (this.running) {
      this.win.requestAnimationFrame(this.loop);
    }
  }

  update() {
    if (!this.isPlaying()) {
      return;
    }

    const observation = this.observe();
    if (!observation) return;

    const previousScore = this.prevObservation ? this.prevObservation.score : (this.prevScore ?? 0);

    if (this.mode === 'train' && observation.score > previousScore) {
      this.bestDistance = observation.foodDistance ?? Infinity;
      this.noProgressSteps = 0;
    }

    if (this.mode === 'train' && observation.foodDistance != null) {
      if (observation.foodDistance < this.bestDistance) {
        this.bestDistance = observation.foodDistance;
        this.noProgressSteps = 0;
      } else {
        this.noProgressSteps += 1;
      }
      if (this.noProgressSteps > this.noProgressLimit) {
        this.endEpisode(this.stallPenalty);
        return;
      }
    }

    const headKey = coordKey(observation.head);
    const headMoved = headKey !== this.lastHeadKey;
    const gameOver = this.isGameOver();

    const currentState = this.extractState(observation);

    if (this.prevState && this.prevActionIndex !== null && headMoved) {
      const reward = this.computeReward(observation, gameOver);
      const done = gameOver;
      this.agent.update(this.prevState, this.prevActionIndex, reward, currentState, done);
      this.prevState = null;
      this.prevActionIndex = null;
      this.prevDistance = observation.foodDistance;
      this.prevScore = observation.score;
    }

    if (gameOver) {
      this.handleEpisodeEnd();
      return;
    }

    if (!this.prevState) {
      const actionIndex = this.agent.chooseAction(currentState);
      const action = TURN_ACTIONS[actionIndex];
      this.executeAction(action, observation);
      this.prevState = currentState;
      this.prevActionIndex = actionIndex;
      this.prevDistance = observation.foodDistance;
      this.prevScore = observation.score;
    }

    this.prevObservation = observation;
    this.lastHeadKey = headKey;
    this.currentEpisodeScore = observation.score;
  }

  computeReward(observation, gameOver) {
    let reward = this.stepPenalty;
    const scoreDiff = observation.score - this.prevScore;
    if (scoreDiff > 0) {
      reward += FOOD_REWARD * scoreDiff;
    }
    if (gameOver) {
      reward += DEATH_PENALTY;
    }
    if (this.prevDistance != null && observation.foodDistance != null) {
      const diff = this.prevDistance - observation.foodDistance;
      if (diff > 0) reward += DIST_SHAPING;
      else if (diff < 0) reward -= DIST_SHAPING;
    }
    return reward;
  }

  executeAction(action, observation) {
    const currentDir = observation.dirVec;
    let targetDir = currentDir;
    if (action === -1) {
      targetDir = this.turnLeft(currentDir);
    } else if (action === 1) {
      targetDir = this.turnRight(currentDir);
    }
    if (!targetDir) return;
    if (targetDir.dx === currentDir.dx && targetDir.dy === currentDir.dy) return;
    const dirEntry = DIRS.find((d) => d.dx === targetDir.dx && d.dy === targetDir.dy);
    if (!dirEntry) return;
    const event = new KeyboardEvent('keydown', {
      key: dirEntry.key,
      code: dirEntry.key,
      bubbles: true
    });
    this.win.dispatchEvent(event);
    this.currentDir = targetDir;
  }

  handleEpisodeEnd() {
    this.endEpisode(DEATH_PENALTY);
  }

  endEpisode(terminalReward = DEATH_PENALTY) {
    if (this.prevState && this.prevActionIndex !== null) {
      this.agent.update(this.prevState, this.prevActionIndex, terminalReward, null, true);
    }
    const episodeScore = this.prevObservation?.score ?? this.currentEpisodeScore ?? 0;
    this.prevState = null;
    this.prevActionIndex = null;
    this.prevObservation = null;
    this.prevDistance = null;
    this.lastHeadKey = null;
    this.noProgressSteps = 0;
    this.bestDistance = Infinity;
    this.currentEpisodeScore = 0;

    if (this.mode === 'train') {
      this.episodeScores.push(episodeScore);
      const windowSize = 50;
      this.scoreWindow.push(episodeScore);
      if (this.scoreWindow.length > windowSize) {
        this.scoreWindow.shift();
      }
      if (this.completedEpisodes % windowSize === 0) {
        const avg = this.scoreWindow.reduce((sum, val) => sum + val, 0) / this.scoreWindow.length;
        let deltaPct = null;
        if (this.lastWindowAverage != null && this.lastWindowAverage !== 0) {
          deltaPct = ((avg - this.lastWindowAverage) / Math.abs(this.lastWindowAverage)) * 100;
        }
        const formattedAvg = avg.toFixed(2);
        const formattedDelta = deltaPct == null ? '' : ` (${deltaPct >= 0 ? '+' : ''}${deltaPct.toFixed(1)}%)`;
        const start = Math.max(1, this.completedEpisodes - windowSize + 1);
        const end = this.completedEpisodes;
        console.info(`[snakeQ] Episodes ${start}-${end}: ${formattedAvg}${formattedDelta}`);
        this.lastWindowAverage = avg;
      }
    }

    this.completedEpisodes += 1;
    if (this.mode === 'train') {
      if (this.autoSaveInterval && (this.completedEpisodes % this.autoSaveInterval === 0)) {
        try {
          this.agent.save();
        } catch (err) {
          console.warn('[SnakeQLearner] auto-save failed', err);
        }
      }
      if (this.completedEpisodes >= this.targetEpisodes) {
        this.agent.save();
        this.stop();
      } else {
        this.agent.decayEpsilon(this.epsilonMin, this.epsilonDecay);
        if (this.alphaDecay && this.alphaDecay !== 1) {
          this.agent.alpha = Math.max(0.05, this.agent.alpha * this.alphaDecay);
        }
        this.resetGame();
      }
    } else {
      this.stop();
    }
  }

  observe() {
    const width = this.canvas.width;
    const height = this.canvas.height;
    if (!width || !height) return null;

    const image = this.ctx.getImageData(0, 0, width, height).data;
    const metrics = [];
    const brightCells = [];

    for (let y = 0; y < this.GRID; y++) {
      metrics[y] = [];
      for (let x = 0; x < this.GRID; x++) {
        const sample = this.sampleCell(image, width, x, y);
        sample.bright = sample.value > 28 && sample.a > 15;
        metrics[y][x] = sample;
        if (sample.bright) brightCells.push({ x, y });
      }
    }
    if (!brightCells.length) return null;

    const clusters = this.buildClusters(metrics);
    if (!clusters.length) return null;

    const snakeCluster = this.identifySnakeCluster(clusters);
    if (!snakeCluster) return null;

    const snake = this.reconstructSnake(snakeCluster, metrics);
    if (!snake) return null;

    const foodCluster = clusters.find((c) => c !== snakeCluster);
    const food = this.pickFoodCell(foodCluster, metrics);

    const score = this.getScore();
    const foodDistance = food ? manhattan(snake.head, food) : null;

    const dirVec = this.currentDirectionFromSnake(snake.body);

    return {
      head: snake.head,
      body: snake.body,
      dirVec,
      food,
      score,
      foodDistance
    };
  }

  extractState(observation) {
    const { head, body, dirVec, food } = observation;
    this.currentDir = { dx: dirVec.dx, dy: dirVec.dy };
    const forward = dirVec;
    const left = this.turnLeft(dirVec);
    const right = this.turnRight(dirVec);

    const occupied = new Set(body.map(coordKey));

    const danger = (vec) => {
      const nx = head.x + vec.dx;
      const ny = head.y + vec.dy;
      if (nx < 0 || ny < 0 || nx >= this.GRID || ny >= this.GRID) return 1;
      return occupied.has(`${nx},${ny}`) ? 1 : 0;
    };

    const dFront = danger(forward);
    const dLeft = danger(left);
    const dRight = danger(right);

    const dirIndex = this.directionIndex(dirVec);

    let foodX = 0;
    let foodY = 0;
    if (food) {
      const foodVec = { x: food.x - head.x, y: food.y - head.y };
      const rightVec = this.turnRight(dirVec);
      const dotForward = foodVec.x * forward.dx + foodVec.y * forward.dy;
      const dotRight = foodVec.x * rightVec.dx + foodVec.y * rightVec.dy;
      foodY = dotForward > 0 ? 1 : dotForward < 0 ? -1 : 0;
      foodX = dotRight > 0 ? 1 : dotRight < 0 ? -1 : 0;
    }

    return {
      dFront,
      dLeft,
      dRight,
      dir: dirIndex,
      foodX,
      foodY
    };
  }

  directionIndex(dir) {
    if (!dir) return 0;
    if (dir.dx === 0 && dir.dy === -1) return 0;
    if (dir.dx === 1 && dir.dy === 0) return 1;
    if (dir.dx === 0 && dir.dy === 1) return 2;
    if (dir.dx === -1 && dir.dy === 0) return 3;
    return 0;
  }

  currentDirectionFromSnake(body) {
    if (!body || body.length < 2) {
      return this.currentDir || { dx: 1, dy: 0 };
    }
    const [head, neck] = body;
    const dx = head.x - neck.x;
    const dy = head.y - neck.y;
    const dir = DIRS.find((d) => d.dx === dx && d.dy === dy);
    if (dir) return { dx: dir.dx, dy: dir.dy };
    return this.currentDir || { dx: 1, dy: 0 };
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
          if (nx < 0 || ny < 0 || nx >= this.GRID || ny >= this.GRID) continue;
          if (!metrics[ny][nx].bright) continue;
          const key = coordKey({ x: nx, y: ny });
          if (visited.has(key)) continue;
          visited.add(key);
          stack.push([nx, ny]);
        }
      }
      return { cells, set };
    };

    for (let y = 0; y < this.GRID; y++) {
      for (let x = 0; x < this.GRID; x++) {
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
    let best = null;
    let bestScore = -1;
    for (const cluster of clusters) {
      if (cluster.cells.length > bestScore) {
        best = cluster;
        bestScore = cluster.cells.length;
      }
    }
    return best;
  }

  reconstructSnake(cluster, metrics) {
    if (!cluster) return null;
    const cells = cluster.cells.map((c) => ({ x: c.x, y: c.y }));
    if (!cells.length) return null;

    const prevHead = this.prevObservation?.head;
    let head = null;
    if (prevHead) {
      let bestDist = Infinity;
      for (const cell of cells) {
        const dist = manhattan(cell, prevHead);
        if (dist < bestDist) {
          bestDist = dist;
          head = cell;
        }
        if (bestDist === 0) break;
      }
    }
    if (!head) {
      let bestValue = -Infinity;
      for (const cell of cells) {
        const value = metrics[cell.y][cell.x].value;
        if (value > bestValue) {
          bestValue = value;
          head = cell;
        }
      }
    }
    if (!head) head = cells[0];
    let tail = head;

    const body = [head];
    const set = new Set(cells.map(coordKey));
    const visited = new Set([coordKey(head)]);
    let current = head;
    let prev = null;
    while (true) {
      let next = null;
      const neighbors = [
        { x: current.x + 1, y: current.y },
        { x: current.x - 1, y: current.y },
        { x: current.x, y: current.y + 1 },
        { x: current.x, y: current.y - 1 }
      ];
      for (const candidate of neighbors) {
        const key = coordKey(candidate);
        if (!set.has(key)) continue;
        if (visited.has(key)) continue;
        next = candidate;
        break;
      }
      if (!next) break;
      body.push(next);
      visited.add(coordKey(next));
      prev = current;
      current = next;
      tail = next;
      if (body.length > cells.length + 2) break;
    }

    return {
      head,
      tail,
      body
    };
  }

  pickFoodCell(cluster, metrics) {
    if (!cluster) return null;
    let brightest = null;
    let bestValue = -Infinity;
    for (const cell of cluster.cells) {
      const value = metrics[cell.y][cell.x].value;
      if (value > bestValue) {
        bestValue = value;
        brightest = { x: cell.x, y: cell.y };
      }
    }
    return brightest;
  }

  turnLeft(dir) {
    const dx = dir.dx !== undefined ? dir.dx : dir.x || 0;
    const dy = dir.dy !== undefined ? dir.dy : dir.y || 0;
    if (dx === 1 && dy === 0) return { dx: 0, dy: -1 };
    if (dx === -1 && dy === 0) return { dx: 0, dy: 1 };
    if (dx === 0 && dy === 1) return { dx: 1, dy: 0 };
    if (dx === 0 && dy === -1) return { dx: -1, dy: 0 };
    return { dx: 1, dy: 0 };
  }

  turnRight(dir) {
    const dx = dir.dx !== undefined ? dir.dx : dir.x || 0;
    const dy = dir.dy !== undefined ? dir.dy : dir.y || 0;
    if (dx === 1 && dy === 0) return { dx: 0, dy: 1 };
    if (dx === -1 && dy === 0) return { dx: 0, dy: -1 };
    if (dx === 0 && dy === 1) return { dx: -1, dy: 0 };
    if (dx === 0 && dy === -1) return { dx: 1, dy: 0 };
    return { dx: 1, dy: 0 };
  }
}

function installQLearner() {
  if (window.__snakeQLearner) return window.__snakeQLearner;
  const learner = new SnakeQLearner(window);
  window.__snakeQLearner = learner;
  console.info('[SnakeQLearner] Ready. Use snakeQ.train({...}) or snakeQ.play({...}).');
  return learner;
}

const snakeQ = installQLearner();
if (typeof window !== 'undefined') {
  window.snakeQ = snakeQ;
}

export { snakeQ };
