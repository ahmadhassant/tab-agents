/* ============================================================================
   tab_nn.js — TD(lambda) value network for the browser build.

   Ports tab_env.encode_pre (124 dims, throw slots left zero, which is what
   tab_td.encode does) and the forward pass of the 124-512-256-128-1 network
   trained by 500,000 self-play games.

   Weights are float16 in base64; float16 was chosen over int8 because int8
   changed the agent's chosen move on 3.75% of decisions, whereas float16
   reproduces the Python argmax on 300/300 sampled decisions.

   Exposed:
     TabNN.load(json)                     -> loads weights
     TabNN.encode(ce, fa, fq, pst, nre, player) -> Float32Array(124)
     TabNN.value(vec)                     -> scalar in [-1, 1]
     TabNN.scoreBoard(ce, fa, fq, pst, nre, player) -> scalar
   ========================================================================== */
var TabNN = (function () {
  var LAYERS = null, IN_DIM = 124;

  var MIRROR = (function () {
    var m = new Int32Array(32);
    for (var p = 0; p < 32; p++) {
      m[p] = p < 8 ? p + 24 : (p < 16 ? p + 8 : (p < 24 ? p - 8 : p - 24));
    }
    return m;
  })();

  function b64ToBytes(b64) {
    var bin = (typeof atob === 'function')
      ? atob(b64) : Buffer.from(b64, 'base64').toString('binary');
    var out = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  /* IEEE-754 half -> float. JS has no native float16 read. */
  function f16to32(h) {
    var s = (h & 0x8000) >> 15, e = (h & 0x7C00) >> 10, f = h & 0x03FF;
    if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    if (e === 0x1F) return f ? NaN : ((s ? -1 : 1) * Infinity);
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
  }

  function decodeWeights(L) {
    var bytes = b64ToBytes(L.q), rows = L.shape[0], cols = L.shape[1];
    var W = new Float32Array(rows * cols);
    if (L.dtype === 'f16') {
      var dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      for (var i = 0; i < rows * cols; i++) W[i] = f16to32(dv.getUint16(i * 2, true));
    } else if (L.dtype === 'f32') {
      var dv2 = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      for (var j = 0; j < rows * cols; j++) W[j] = dv2.getFloat32(j * 4, true);
    } else { /* i8, per-row scale */
      for (var r = 0; r < rows; r++) {
        var sc = L.scale[r];
        for (var c = 0; c < cols; c++) {
          var v = bytes[r * cols + c];
          if (v > 127) v -= 256;
          W[r * cols + c] = v * sc;
        }
      }
    }
    return { W: W, b: Float32Array.from(L.b), rows: rows, cols: cols };
  }

  function load(json) {
    IN_DIM = json.in_dim;
    LAYERS = json.layers.map(decodeWeights);
    return LAYERS.length;
  }

  function value(x) {
    var a = x;
    for (var li = 0; li < LAYERS.length; li++) {
      var L = LAYERS[li], out = new Float32Array(L.rows);
      for (var r = 0; r < L.rows; r++) {
        var s = L.b[r], off = r * L.cols;
        for (var c = 0; c < L.cols; c++) s += L.W[off + c] * a[c];
        out[r] = (li < LAYERS.length - 1) ? (s > 0 ? s : 0) : Math.tanh(s);
      }
      a = out;
    }
    return a[0];
  }

  /* --- encode_pre, throw slots zero ------------------------------------- */
  function encode(ce, fa, fq, pst, nre, player) {
    var v = new Float32Array(124), i, k;
    var opp = -player;
    var faP = fa[player] || {}, faO = fa[opp] || {};
    var fqP = fq[player] || [], fqO = fq[opp] || [];

    /* [0:32] mirrored cells / 8 */
    if (player === 1) { for (i = 0; i < 32; i++) v[i] = ce[i] / 8.0; }
    else { for (i = 0; i < 32; i++) v[i] = (ce[MIRROR[i]] * -1) / 8.0; }

    /* [32:40] own queue ranks, [40:48] opponent queue ranks */
    var ownSlots = [], oppSlots = [];
    for (i = 0; i < 8; i++) {
      ownSlots.push(player === 1 ? i : 24 + i);
      oppSlots.push(player === 1 ? 24 + i : i);
    }
    function qfeat(q, faX, slots, base) {
      var rank = {};
      for (var t = 0; t < q.length; t++) rank[q[t]] = q.length - t;
      for (var s = 0; s < 8; s++) {
        var slot = slots[s];
        if ((faX[slot] || 0) > 0 && rank[slot] !== undefined) {
          v[base + s] = rank[slot] / 8.0;
        }
      }
    }
    qfeat(fqP, faP, ownSlots, 32);
    qfeat(fqO, faO, oppSlots, 40);

    /* [48:50] frozen counts */
    v[48] = fqP.length / 8.0;
    v[49] = fqO.length / 8.0;

    /* [50:52] allFreed flags */
    v[50] = fqP.length === 0 ? 1 : 0;
    v[51] = fqO.length === 0 ? 1 : 0;

    /* [52:84] no-reentry taint, mirrored for P2 */
    var nrP = nre[player] || {};
    for (var key in nrP) {
      if (!Object.prototype.hasOwnProperty.call(nrP, key)) continue;
      var stack = nrP[key], any = false;
      for (var z = 0; z < stack.length; z++) if (stack[z]) { any = true; break; }
      if (!any) continue;
      var pos = parseInt(key, 10);
      var mp = (player === 1) ? pos : MIRROR[pos];
      if (mp >= 0 && mp < 32) v[52 + mp] = 1;
    }

    /* [84:86] player_started */
    v[84] = pst[player] ? 1 : 0;
    v[85] = pst[opp] ? 1 : 0;

    /* [86:94] hasActiveElsewhere per own frozen slot */
    var hs = player === 1 ? 0 : 24, he = player === 1 ? 8 : 32, active = false;
    for (i = hs; i < he; i++) {
      if (ce[i] * player > 0 && Math.abs(ce[i]) > (faP[i] || 0)) { active = true; break; }
    }
    if (!active) {
      for (i = 8; i < 24; i++) if (ce[i] * player > 0) { active = true; break; }
    }
    if (active) {
      for (k = 0; k < 8; k++) if ((faP[ownSlots[k]] || 0) > 0) v[86 + k] = 1;
    }

    /* [94:124] throw one-hot — left zero: this is a state value */
    return v;
  }

  function scoreBoard(ce, fa, fq, pst, nre, player) {
    return value(encode(ce, fa, fq, pst, nre, player));
  }

  return { load: load, encode: encode, value: value, scoreBoard: scoreBoard,
             ready: function () { return LAYERS !== null; } };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = TabNN;
