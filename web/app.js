/**
 * Chess-AI Web Client
 * State Management, Interactive SVG Chessboard, Audio Synthesizer & API Bridge.
 */

// High-Definition Staunton Chess SVG Vector Paths
const PIECE_SVGS = {
  // White Pieces
  P: `<svg viewBox="0 0 45 45"><path d="M22.5 9c-2.21 0-4 1.79-4 4 0 .89.29 1.71.78 2.38C17.33 16.5 16 18.59 16 21c0 2.03.94 3.84 2.41 5.03-3 1.06-7.41 5.55-7.41 13.47h23c0-7.92-4.41-12.41-7.41-13.47 1.47-1.19 2.41-3 2.41-5.03 0-2.41-1.33-4.5-3.28-5.62.49-.67.78-1.49.78-2.38 0-2.21-1.79-4-4-4z" fill="#ffffff" stroke="#1e293b" stroke-width="1.6" stroke-linecap="round"/></svg>`,
  N: `<svg viewBox="0 0 45 45"><g fill="none" fill-rule="evenodd" stroke="#1e293b" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M22 10c10.5 1 16.5 8 16 29H15c0-9 10-6.5 8-21" fill="#ffffff"/><path d="M24 18c.38 2.91-5.55 7.37-8 9-3 2-2.82 4.34-5 4-1.042-.94 1.41-3.04 0-3-1 0 .19 1.23-1 2-1 0-4.003 1-4-4 0-2 6-12 6-12s1.89-1.9 2-3.5c-.73-.994-.5-2-.5-3 1-1 3 2.5 3 2.5l2 0s.78-1.992 2.5-3c1 0 1 3 1 3z" fill="#ffffff"/><circle cx="14" cy="14.5" r="1.2" fill="#1e293b"/></g></svg>`,
  B: `<svg viewBox="0 0 45 45"><g fill="none" fill-rule="evenodd" stroke="#1e293b" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><g fill="#ffffff" stroke-linecap="butt"><path d="M9 36c3.39-.97 10.11.43 13.5-2 3.39 2.43 10.11 1.03 13.5 2 0 0 1.65.54 3 2-.68.97-1.65.99-3 .5-3.39-.97-10.11.46-13.5-1-3.39 1.46-10.11.03-13.5 1-1.354.49-2.323.47-3-.5 1.354-1.94 3-2 3-2z"/><path d="M15 32c2.5 2.5 12.5 2.5 15 0 .5-1.5 0-2 0-2 0-2.5-2.5-4-2.5-4 5.5-1.5 6-11.5-5-15.5-11 4-10.5 14-5 15.5 0 0-2.5 1.5-2.5 4 0 0-.5.5 0 2z"/><path d="M25 8a2.5 2.5 0 1 1-5 0 2.5 2.5 0 1 1 5 0z"/></g><path d="M17.5 26h10M15 30h15M22.5 15.5v5M20 18h5" stroke="#1e293b"/></g></svg>`,
  R: `<svg viewBox="0 0 45 45"><g fill="#ffffff" fill-rule="evenodd" stroke="#1e293b" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M9 39h27v-3H9v3zM12 36v-4h21v4H12zM11 14V9h4v2h5V9h5v2h5V9h4v5" stroke-linecap="butt"/><path d="M34 14l-3 3H14l-3-3"/><path d="M31 17v12.5H14V17"/><path d="M31 29.5l1.5 2.5h-20l1.5-2.5"/><path d="M11 14h23"/></g></svg>`,
  Q: `<svg viewBox="0 0 45 45"><g fill="#ffffff" fill-rule="evenodd" stroke="#1e293b" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M8 12a2 2 0 1 1-4 0 2 2 0 1 1 4 0zM24.5 7.5a2 2 0 1 1-4 0 2 2 0 1 1 4 0zM41 12a2 2 0 1 1-4 0 2 2 0 1 1 4 0zM11 20a2 2 0 1 1-4 0 2 2 0 1 1 4 0zM38 20a2 2 0 1 1-4 0 2 2 0 1 1 4 0z"/><path d="M9 26c8.5-1.5 21-1.5 27 0l2-12-7 11V11l-5.5 13.5-3-15-3 15-5.5-13.5V25l-7-11 2 12z"/><path d="M9 26c0 2 1.5 2 2.5 4 2.5 2 5 2.5 11 2.5s8.5-.5 11-2.5c1-2 2.5-2 2.5-4-8.5-1.5-18.5-1.5-27 0z"/><path d="M11 30c3.5 2.5 6 2.5 11.5 2.5s8-.5 11.5-2.5M11.5 34c2.5 1.5 5.5 1.5 11 1.5s8.5 0 11-1.5M12 38c2.5 1 5.5 1 10.5 1s8 0 10.5-1"/></g></svg>`,
  K: `<svg viewBox="0 0 45 45"><g fill="none" fill-rule="evenodd" stroke="#1e293b" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M22.5 11.63V6M20 8h5" stroke="#1e293b"/><path d="M22.5 25s4.5-7.5 3-10.5c0 0-1-2.5-3-2.5s-3 2.5-3 2.5c-1.5 3 3 10.5 3 10.5" fill="#ffffff" stroke="#1e293b"/><path d="M11.5 37c5.5 3.5 15.5 3.5 21 0v-7s9-4.5 6-10.5c-4-1-1 8-6 8-3-4-7.5-3-10.5-3s-7.5-1-10.5 3c-5 0-2-9-6-8-3 6 6 10.5 6 10.5v7z" fill="#ffffff"/><path d="M20 28h5M18 32h9M16 36h13" stroke="#1e293b"/></g></svg>`,

  // Black Pieces
  p: `<svg viewBox="0 0 45 45"><path d="M22.5 9c-2.21 0-4 1.79-4 4 0 .89.29 1.71.78 2.38C17.33 16.5 16 18.59 16 21c0 2.03.94 3.84 2.41 5.03-3 1.06-7.41 5.55-7.41 13.47h23c0-7.92-4.41-12.41-7.41-13.47 1.47-1.19 2.41-3 2.41-5.03 0-2.41-1.33-4.5-3.28-5.62.49-.67.78-1.49.78-2.38 0-2.21-1.79-4-4-4z" fill="#1e293b" stroke="#ffffff" stroke-width="1.6" stroke-linecap="round"/></svg>`,
  n: `<svg viewBox="0 0 45 45"><g fill="none" fill-rule="evenodd" stroke="#ffffff" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M22 10c10.5 1 16.5 8 16 29H15c0-9 10-6.5 8-21" fill="#1e293b"/><path d="M24 18c.38 2.91-5.55 7.37-8 9-3 2-2.82 4.34-5 4-1.042-.94 1.41-3.04 0-3-1 0 .19 1.23-1 2-1 0-4.003 1-4-4 0-2 6-12 6-12s1.89-1.9 2-3.5c-.73-.994-.5-2-.5-3 1-1 3 2.5 3 2.5l2 0s.78-1.992 2.5-3c1 0 1 3 1 3z" fill="#1e293b"/><circle cx="14" cy="14.5" r="1.2" fill="#ffffff"/></g></svg>`,
  b: `<svg viewBox="0 0 45 45"><g fill="none" fill-rule="evenodd" stroke="#ffffff" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><g fill="#1e293b" stroke-linecap="butt"><path d="M9 36c3.39-.97 10.11.43 13.5-2 3.39 2.43 10.11 1.03 13.5 2 0 0 1.65.54 3 2-.68.97-1.65.99-3 .5-3.39-.97-10.11.46-13.5-1-3.39 1.46-10.11.03-13.5 1-1.354.49-2.323.47-3-.5 1.354-1.94 3-2 3-2z"/><path d="M15 32c2.5 2.5 12.5 2.5 15 0 .5-1.5 0-2 0-2 0-2.5-2.5-4-2.5-4 5.5-1.5 6-11.5-5-15.5-11 4-10.5 14-5 15.5 0 0-2.5 1.5-2.5 4 0 0-.5.5 0 2z"/><path d="M25 8a2.5 2.5 0 1 1-5 0 2.5 2.5 0 1 1 5 0z"/></g><path d="M17.5 26h10M15 30h15M22.5 15.5v5M20 18h5" stroke="#ffffff"/></g></svg>`,
  r: `<svg viewBox="0 0 45 45"><g fill="#1e293b" fill-rule="evenodd" stroke="#ffffff" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M9 39h27v-3H9v3zM12 36v-4h21v4H12zM11 14V9h4v2h5V9h5v2h5V9h4v5" stroke-linecap="butt"/><path d="M34 14l-3 3H14l-3-3"/><path d="M31 17v12.5H14V17"/><path d="M31 29.5l1.5 2.5h-20l1.5-2.5"/><path d="M11 14h23"/></g></svg>`,
  q: `<svg viewBox="0 0 45 45"><g fill="#1e293b" fill-rule="evenodd" stroke="#ffffff" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M8 12a2 2 0 1 1-4 0 2 2 0 1 1 4 0zM24.5 7.5a2 2 0 1 1-4 0 2 2 0 1 1 4 0zM41 12a2 2 0 1 1-4 0 2 2 0 1 1 4 0zM11 20a2 2 0 1 1-4 0 2 2 0 1 1 4 0zM38 20a2 2 0 1 1-4 0 2 2 0 1 1 4 0z"/><path d="M9 26c8.5-1.5 21-1.5 27 0l2-12-7 11V11l-5.5 13.5-3-15-3 15-5.5-13.5V25l-7-11 2 12z"/><path d="M9 26c0 2 1.5 2 2.5 4 2.5 2 5 2.5 11 2.5s8.5-.5 11-2.5c1-2 2.5-2 2.5-4-8.5-1.5-18.5-1.5-27 0z"/><path d="M11 30c3.5 2.5 6 2.5 11.5 2.5s8-.5 11.5-2.5M11.5 34c2.5 1.5 5.5 1.5 11 1.5s8.5 0 11-1.5M12 38c2.5 1 5.5 1 10.5 1s8 0 10.5-1"/></g></svg>`,
  k: `<svg viewBox="0 0 45 45"><g fill="none" fill-rule="evenodd" stroke="#ffffff" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M22.5 11.63V6M20 8h5" stroke="#ffffff"/><path d="M22.5 25s4.5-7.5 3-10.5c0 0-1-2.5-3-2.5s-3 2.5-3 2.5c-1.5 3 3 10.5 3 10.5" fill="#1e293b" stroke="#ffffff"/><path d="M11.5 37c5.5 3.5 15.5 3.5 21 0v-7s9-4.5 6-10.5c-4-1-1 8-6 8-3-4-7.5-3-10.5-3s-7.5-1-10.5 3c-5 0-2-9-6-8-3 6 6 10.5 6 10.5v7z" fill="#1e293b"/><path d="M20 28h5M18 32h9M16 36h13" stroke="#ffffff"/></g></svg>`,
};

// Sound Effects Synthesizer using Web Audio API (Soft & Subtle)
class ChessSoundFx {
  constructor() {
    this.ctx = null;
    this.muted = false;
  }

  init() {
    if (!this.ctx) {
      this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    }
  }

  toggleMute() {
    this.muted = !this.muted;
    const btn = document.getElementById("btnSoundToggle");
    if (btn) {
      btn.textContent = this.muted ? "🔇 Muted" : "🔊 Sound";
      btn.classList.toggle("muted", this.muted);
    }
    return this.muted;
  }

  playMove() {
    if (this.muted) return;
    try {
      this.init();
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(300, this.ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(
        120,
        this.ctx.currentTime + 0.05,
      );
      gain.gain.setValueAtTime(0.04, this.ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(
        0.001,
        this.ctx.currentTime + 0.05,
      );
      osc.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start();
      osc.stop(this.ctx.currentTime + 0.05);
    } catch (e) {}
  }

  playCapture() {
    if (this.muted) return;
    try {
      this.init();
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = "triangle";
      osc.frequency.setValueAtTime(380, this.ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(
        90,
        this.ctx.currentTime + 0.07,
      );
      gain.gain.setValueAtTime(0.05, this.ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(
        0.001,
        this.ctx.currentTime + 0.07,
      );
      osc.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start();
      osc.stop(this.ctx.currentTime + 0.07);
    } catch (e) {}
  }

  playCheck() {
    if (this.muted) return;
    try {
      this.init();
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(520, this.ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(
        350,
        this.ctx.currentTime + 0.1,
      );
      gain.gain.setValueAtTime(0.05, this.ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.1);
      osc.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start();
      osc.stop(this.ctx.currentTime + 0.1);
    } catch (e) {}
  }
}

const soundFx = new ChessSoundFx();

// Game State Management
class ChessApp {
  constructor() {
    this.boardEl = document.getElementById("chessboard");
    this.evalFill = document.getElementById("evalBarFill");
    this.evalText = document.getElementById("evalIndicatorText");
    this.evalTop = document.getElementById("evalScoreTop");
    this.evalBottom = document.getElementById("evalScoreBottom");

    this.currentFen =
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    this.isFlipped = false;
    this.playerColor = "w"; // 'w' = White, 'b' = Black, 'self' = Self-play
    this.selectedSquare = null;
    this.legalMoves = [];
    this.lastMove = null;
    this.isThinking = false;

    this.moveHistory = [];
    this.pendingPromotion = null;

    this.initUI();
    this.initStreamlitListener();
    this.fetchSystemStatus();
    this.setPlayerMode("w");
  }

  initUI() {
    // Engine config
    document.getElementById("engineSelect").addEventListener("change", (e) => {
      const isMcts = e.target.value === "mcts";
      document.getElementById("depthGroup").classList.toggle("hidden", isMcts);
      document.getElementById("simsGroup").classList.toggle("hidden", !isMcts);
    });

    document.getElementById("depthSlider").addEventListener("input", (e) => {
      document.getElementById("depthVal").textContent = e.target.value;
    });

    document.getElementById("simsSlider").addEventListener("input", (e) => {
      document.getElementById("simsVal").textContent = e.target.value;
    });

    // Player mode buttons
    document
      .getElementById("btnPlayWhite")
      .addEventListener("click", () => this.setPlayerMode("w"));
    document
      .getElementById("btnPlayBlack")
      .addEventListener("click", () => this.setPlayerMode("b"));
    document
      .getElementById("btnSelfPlay")
      .addEventListener("click", () => this.setPlayerMode("self"));

    // Sound toggle
    const soundBtn = document.getElementById("btnSoundToggle");
    if (soundBtn) {
      soundBtn.addEventListener("click", () => soundFx.toggleMute());
    }

    // Actions
    document
      .getElementById("btnNewGame")
      .addEventListener("click", () => this.resetGame());
    document
      .getElementById("btnUndo")
      .addEventListener("click", () => this.undoMove());
    document.getElementById("btnFlip").addEventListener("click", () => {
      this.isFlipped = !this.isFlipped;
      this.renderBoard();
    });
    document
      .getElementById("btnExportPGN")
      .addEventListener("click", () => this.exportPGN());
  }

  async fetchSystemStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      if (data.device) {
        document.getElementById("deviceBadge").textContent =
          `${data.device.toUpperCase()}: Active`;
      }
      if (data.model) {
        document.getElementById("modelBadge").textContent = data.model;
      }
    } catch (err) {
      document.getElementById("deviceBadge").textContent = "CUDA/CPU: Active";
      document.getElementById("modelBadge").textContent = "ChessResNet v3";
    }
  }

  setPlayerMode(color) {
    this.playerColor = color;
    document
      .getElementById("btnPlayWhite")
      .classList.toggle("active", color === "w");
    document
      .getElementById("btnPlayBlack")
      .classList.toggle("active", color === "b");
    document
      .getElementById("btnSelfPlay")
      .classList.toggle("active", color === "self");

    if (color === "b") {
      this.isFlipped = true;
      document.getElementById("topPlayerName").textContent =
        "AI Engine (White)";
      document.getElementById("bottomPlayerName").textContent = "You (Black)";
    } else if (color === "w") {
      this.isFlipped = false;
      document.getElementById("topPlayerName").textContent =
        "AI Engine (Black)";
      document.getElementById("bottomPlayerName").textContent = "You (White)";
    } else {
      document.getElementById("topPlayerName").textContent = "AI Engine 1";
      document.getElementById("bottomPlayerName").textContent = "AI Engine 2";
    }

    this.resetGame();
  }

  resetGame() {
    this.currentFen =
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    this.moveHistory = [];
    this.selectedSquare = null;
    this.legalMoves = [];
    this.lastMove = null;
    this.isThinking = false;
    this.updateMoveHistoryTable();
    this.updateEvalBar(0.0);
    this.renderBoard();

    // If human is Black or self-play, trigger AI's first move
    const turn = this.getTurn();
    if (
      (this.playerColor === "b" && turn === "w") ||
      this.playerColor === "self"
    ) {
      this.triggerAiMove();
    }
  }

  getTurn() {
    return this.currentFen.split(" ")[1] || "w";
  }

  renderBoard() {
    this.boardEl.innerHTML = "";
    const ranks = this.isFlipped
      ? [0, 1, 2, 3, 4, 5, 6, 7]
      : [7, 6, 5, 4, 3, 2, 1, 0];
    const files = this.isFlipped
      ? [7, 6, 5, 4, 3, 2, 1, 0]
      : [0, 1, 2, 3, 4, 5, 6, 7];

    const boardMatrix = this.fenToMatrix(this.currentFen);
    const turn = this.getTurn();

    // Turn badge update
    const isHumanTurn =
      this.playerColor === turn && this.playerColor !== "self";
    document
      .getElementById("bottomPlayerStatus")
      .classList.toggle("active-turn", isHumanTurn);
    document
      .getElementById("topPlayerStatus")
      .classList.toggle("active-turn", !isHumanTurn);
    document.getElementById("bottomPlayerStatus").textContent = isHumanTurn
      ? "Your Turn"
      : "Waiting";
    document.getElementById("topPlayerStatus").textContent = !isHumanTurn
      ? "Thinking"
      : "Waiting";

    for (let r of ranks) {
      for (let f of files) {
        const sqName = `${String.fromCharCode(97 + f)}${r + 1}`;
        const sqEl = document.createElement("div");
        sqEl.className = `square ${(r + f) % 2 === 0 ? "dark" : "light"}`;
        sqEl.dataset.square = sqName;

        // Coordinate labels
        if (f === (this.isFlipped ? 7 : 0)) {
          const rankLabel = document.createElement("span");
          rankLabel.className = "square-coord rank";
          rankLabel.textContent = `${r + 1}`;
          sqEl.appendChild(rankLabel);
        }
        if (r === (this.isFlipped ? 7 : 0)) {
          const fileLabel = document.createElement("span");
          fileLabel.className = "square-coord file";
          fileLabel.textContent = String.fromCharCode(97 + f);
          sqEl.appendChild(fileLabel);
        }

        // Highlight selected
        if (this.selectedSquare === sqName) {
          sqEl.classList.add("selected");
        }

        // Highlight last move
        if (
          this.lastMove &&
          (this.lastMove.from === sqName || this.lastMove.to === sqName)
        ) {
          sqEl.classList.add("highlight-last");
        }

        // Piece rendering
        const piece = boardMatrix[7 - r][f];
        if (piece) {
          const pieceEl = document.createElement("div");
          pieceEl.className = "piece";
          pieceEl.innerHTML = PIECE_SVGS[piece] || "";
          sqEl.appendChild(pieceEl);
        }

        // Legal move indicators (appended on top of piece overlay)
        const isLegal = this.legalMoves.find((m) => m.to === sqName);
        if (isLegal) {
          const isCapture = isLegal.isCapture || piece !== "";
          const dot = document.createElement("div");
          dot.className = isCapture ? "legal-capture-ring" : "legal-dot";
          sqEl.appendChild(dot);
        }

        sqEl.addEventListener("click", () => this.handleSquareClick(sqName));
        this.boardEl.appendChild(sqEl);
      }
    }
  }

  fenToMatrix(fen) {
    const rows = fen.split(" ")[0].split("/");
    const matrix = [];
    for (let r of rows) {
      const row = [];
      for (let c of r) {
        if (!isNaN(c)) {
          for (let i = 0; i < parseInt(c); i++) row.push("");
        } else {
          row.push(c);
        }
      }
      matrix.push(row);
    }
    return matrix;
  }

  async handleSquareClick(sqName) {
    if (this.isThinking) return;

    const turn = this.getTurn();
    const isHumanTurn =
      this.playerColor === turn && this.playerColor !== "self";
    if (!isHumanTurn) return;

    // If square clicked is a legal destination move
    const targetMove = this.legalMoves.find((m) => m.to === sqName);
    if (this.selectedSquare && targetMove) {
      if (targetMove.isPromotion) {
        this.promptPromotion(this.selectedSquare, sqName);
        return;
      }
      this.executeMove(this.selectedSquare, sqName);
      return;
    }

    // Select piece
    const boardMatrix = this.fenToMatrix(this.currentFen);
    const f = sqName.charCodeAt(0) - 97;
    const r = parseInt(sqName[1]) - 1;
    const piece = boardMatrix[7 - r][f];

    if (piece && this.isPieceColor(piece, turn)) {
      this.selectedSquare = sqName;
      this.computeLegalMoves(sqName);
      this.renderBoard(); // INSTANT VISUAL HIGHLIGHT & LEGAL MOVE DOTS (0ms latency!)
    } else {
      this.selectedSquare = null;
      this.legalMoves = [];
      this.renderBoard();
    }
  }

  isPieceColor(piece, color) {
    return color === "w"
      ? piece === piece.toUpperCase()
      : piece === piece.toLowerCase();
  }

  computeLegalMoves(sq) {
    if (typeof Chess !== "undefined") {
      try {
        const game = new Chess(this.currentFen);
        const moves = game.moves({ square: sq, verbose: true });
        this.legalMoves = moves.map((m) => ({
          to: m.to,
          uci: `${m.from}${m.to}${m.promotion || ""}`,
          isCapture: m.flags.includes("c") || m.flags.includes("e"),
          isPromotion: m.flags.includes("p"),
        }));
        return;
      } catch (err) {
        console.warn("Client chess.js legal moves error:", err);
      }
    }
    this.fetchLegalMoves(sq);
  }

  async fetchLegalMoves(sq) {
    try {
      const res = await fetch("/api/legal_moves", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fen: this.currentFen, square: sq }),
      });
      const data = await res.json();
      this.legalMoves = data.moves || [];
      this.renderBoard();
    } catch (err) {
      this.legalMoves = [];
    }
  }

  promptPromotion(fromSq, toSq) {
    this.pendingPromotion = { from: fromSq, to: toSq };
    const modal = document.getElementById("promotionModal");
    const choices = document.getElementById("promotionChoices");
    choices.innerHTML = "";

    const turn = this.getTurn();
    const pieces = turn === "w" ? ["Q", "R", "B", "N"] : ["q", "r", "b", "n"];

    for (let p of pieces) {
      const btn = document.createElement("div");
      btn.className = "promo-btn";
      btn.innerHTML = PIECE_SVGS[p];
      btn.onclick = () => {
        modal.classList.remove("active");
        this.executeMove(fromSq, toSq, p.toLowerCase());
      };
      choices.appendChild(btn);
    }
    modal.classList.add("active");
  }

  executeMove(fromSq, toSq, promo = null) {
    const moveUci = `${fromSq}${toSq}${promo || ""}`;
    this.selectedSquare = null;
    this.legalMoves = [];

    if (typeof Chess !== "undefined") {
      try {
        const game = new Chess(this.currentFen);
        const moveObj = game.move({ from: fromSq, to: toSq, promotion: promo });
        if (moveObj) {
          this.currentFen = game.fen();
          this.lastMove = { from: fromSq, to: toSq };
          this.moveHistory.push({
            san: moveObj.san,
            uci: moveUci,
            isCapture: !!moveObj.captured,
          });

          this.updateMoveHistoryTable();
          this.renderBoard(); // INSTANT VISUAL UPDATE (0ms delay!)

          if (moveObj.captured) soundFx.playCapture();
          else soundFx.playMove();

          if (game.in_check()) soundFx.playCheck();

          if (game.game_over()) {
            let res = "1/2-1/2";
            if (game.in_checkmate()) res = game.turn() === "w" ? "0-1" : "1-0";
            this.handleGameOver(res);
            return;
          }

          // Trigger AI move
          if (this.playerColor !== "self") {
            this.triggerAiMove();
          } else {
            setTimeout(() => this.triggerAiMove(), 500);
          }
          return;
        }
      } catch (err) {
        console.warn("Client chess.js move failed, trying server:", err);
      }
    }

    this.serverExecuteMove(fromSq, toSq, promo);
  }

  async serverExecuteMove(fromSq, toSq, promo = null) {
    const moveUci = `${fromSq}${toSq}${promo || ""}`;
    try {
      const res = await fetch("/api/apply_move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fen: this.currentFen, uci: moveUci }),
      });
      const data = await res.json();
      if (data.error) return;

      this.currentFen = data.fen;
      this.lastMove = { from: fromSq, to: toSq };
      this.moveHistory.push({
        san: data.san,
        uci: moveUci,
        isCapture: data.isCapture,
      });

      this.updateMoveHistoryTable();
      this.updateEvalBar(data.eval);
      this.renderBoard();

      if (data.isCapture) soundFx.playCapture();
      else soundFx.playMove();

      if (data.isCheck) soundFx.playCheck();

      if (data.isGameOver) {
        this.handleGameOver(data.result);
        return;
      }

      if (this.playerColor !== "self") {
        this.triggerAiMove();
      } else {
        setTimeout(() => this.triggerAiMove(), 500);
      }
    } catch (err) {
      console.error("Move application error:", err);
    }
  }

  initStreamlitListener() {
    window.addEventListener("message", (event) => {
      if (event.data && event.data.type === "streamlit:render") {
        const args = event.data.args;
        if (
          args &&
          args.ai_result &&
          args.ai_result.uci &&
          args.ai_result.fen !== this.currentFen
        ) {
          this.applyAiMoveData(args.ai_result);
        }
      }
    });
  }

  applyAiMoveData(data) {
    if (!data || !data.uci) return;
    const fromSq = data.uci.slice(0, 2);
    const toSq = data.uci.slice(2, 4);

    this.currentFen = data.fen;
    this.lastMove = { from: fromSq, to: toSq };
    this.moveHistory.push({
      san: data.san,
      uci: data.uci,
      isCapture: data.isCapture,
    });

    this.updateMoveHistoryTable();
    this.updateEvalBar(data.eval);
    this.renderBoard();

    if (data.isCapture) soundFx.playCapture();
    else soundFx.playMove();

    if (data.isCheck) soundFx.playCheck();

    const calcTime = data.calcTimeMs ? `${data.calcTimeMs}ms` : "0ms";
    document.getElementById("statTime").textContent = calcTime;
    const evalVal = typeof data.eval === "number" ? data.eval : 0.0;
    document.getElementById("statEval").textContent =
      `${evalVal > 0 ? "+" : ""}${evalVal.toFixed(2)}`;

    this.isThinking = false;
    document.getElementById("statStatus").textContent = "Live Match";

    if (data.isGameOver) {
      this.handleGameOver(data.result);
    } else if (this.playerColor === "self") {
      setTimeout(() => this.triggerAiMove(), 400);
    }
  }

  async triggerAiMove() {
    this.isThinking = true;
    document.getElementById("statStatus").textContent = "AI Calculating...";

    const engineType = document.getElementById("engineSelect").value;
    const depth = parseInt(document.getElementById("depthSlider").value);
    const sims = parseInt(document.getElementById("simsSlider").value);

    // Send Streamlit Component message if embedded in Streamlit
    if (window.parent && window.parent !== window) {
      try {
        window.parent.postMessage(
          {
            isStreamlitMessage: true,
            type: "streamlit:setComponentValue",
            value: {
              action: "ai_move",
              fen: this.currentFen,
              engine: engineType,
              depth: depth,
              simulations: sims,
            },
          },
          "*"
        );
      } catch (e) {}
    }

    let data = null;
    const startTime = performance.now();

    // 1. Primary: Dual-Head PyTorch ResNet Model (models/chess_model_v3.pth)
    try {
      const res = await fetch("http://127.0.0.1:8000/api/move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          fen: this.currentFen,
          engine: engineType,
          depth: depth,
          simulations: sims,
        }),
      });
      if (res.ok) {
        data = await res.json();
      }
    } catch (err) {
      console.warn("PyTorch server endpoint unreachable:", err);
    }

    if (data && data.uci) {
      this.applyAiMoveData(data);
    } else {
      this.isThinking = false;
      document.getElementById("statStatus").textContent = "Live Match";
    }
  }

  async undoMove() {
    if (this.moveHistory.length === 0 || this.isThinking) return;

    const steps =
      this.playerColor === "self" || this.moveHistory.length === 1 ? 1 : 2;
    for (let i = 0; i < steps; i++) {
      if (this.moveHistory.length > 0) {
        this.moveHistory.pop();
      }
    }

    const remainingUcis = this.moveHistory.map((m) => m.uci);

    try {
      const res = await fetch("/api/undo", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ moves: remainingUcis }),
      });
      const data = await res.json();
      this.currentFen = data.fen;
      this.selectedSquare = null;
      this.legalMoves = [];
      this.lastMove = data.lastMove || null;
      this.updateMoveHistoryTable();
      this.updateEvalBar(data.eval);
      this.renderBoard();
    } catch (err) {
      console.error("Undo error:", err);
    }
  }

  updateEvalBar(score) {
    if (score === undefined || isNaN(score)) score = 0.0;
    const clamped = Math.max(-1.0, Math.min(1.0, score));
    const percent = ((clamped + 1.0) / 2.0) * 100;
    this.evalFill.style.height = `${percent}%`;
    const sign = score > 0 ? "+" : "";
    const scoreStr = `${sign}${score.toFixed(1)}`;
    this.evalText.textContent = `${sign}${score.toFixed(2)}`;
    if (score >= 0) {
      this.evalTop.textContent = scoreStr;
      this.evalBottom.textContent = "";
    } else {
      this.evalTop.textContent = "";
      this.evalBottom.textContent = scoreStr;
    }
  }

  updateMoveHistoryTable() {
    const tbody = document.getElementById("moveHistoryBody");
    tbody.innerHTML = "";

    for (let i = 0; i < this.moveHistory.length; i += 2) {
      const row = document.createElement("tr");
      const numCell = document.createElement("td");
      numCell.textContent = `${Math.floor(i / 2) + 1}.`;

      const whiteCell = document.createElement("td");
      whiteCell.textContent = this.moveHistory[i]
        ? this.moveHistory[i].san
        : "";

      const blackCell = document.createElement("td");
      blackCell.textContent = this.moveHistory[i + 1]
        ? this.moveHistory[i + 1].san
        : "";

      row.appendChild(numCell);
      row.appendChild(whiteCell);
      row.appendChild(blackCell);
      tbody.appendChild(row);
    }
    tbody.parentElement.parentElement.scrollTop =
      tbody.parentElement.parentElement.scrollHeight;
  }

  handleGameOver(result) {
    document.getElementById("statStatus").textContent = `Game Over: ${result}`;
    alert(`🏆 Game Over!\nResult: ${result}`);
  }

  exportPGN() {
    let pgn = `[Event "Chess-AI Arena Match"]\n[Date "${new Date().toISOString().slice(0, 10)}"]\n\n`;
    for (let i = 0; i < this.moveHistory.length; i += 2) {
      const num = Math.floor(i / 2) + 1;
      const w = this.moveHistory[i].san;
      const b = this.moveHistory[i + 1] ? this.moveHistory[i + 1].san : "";
      pgn += `${num}. ${w} ${b} `;
    }
    const blob = new Blob([pgn], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `chess_match_${Date.now()}.pgn`;
    a.click();
    URL.revokeObjectURL(url);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  window.chessApp = new ChessApp();
});
