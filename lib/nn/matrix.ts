// Minimal dependency-free matrix/vector math. Rows are samples, columns are
// features/units. No linear-algebra library is involved, so every operation a
// forward/backward pass needs is spelled out explicitly here.

export type Matrix = number[][]

export function zeros(rows: number, cols: number): Matrix {
  return Array.from({ length: rows }, () => new Array(cols).fill(0))
}

// He initialization (good default for ReLU; a reasonable one for tanh/sigmoid too).
export function randomMatrix(rows: number, cols: number, fanIn: number): Matrix {
  const scale = Math.sqrt(2 / fanIn)
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => (Math.random() * 2 - 1) * scale),
  )
}

export function multiply(a: Matrix, b: Matrix): Matrix {
  const rowsA = a.length
  const colsA = a[0].length
  const colsB = b[0].length
  const out = zeros(rowsA, colsB)
  for (let i = 0; i < rowsA; i++) {
    for (let k = 0; k < colsA; k++) {
      const aik = a[i][k]
      if (aik === 0) continue
      const bRow = b[k]
      const outRow = out[i]
      for (let j = 0; j < colsB; j++) {
        outRow[j] += aik * bRow[j]
      }
    }
  }
  return out
}

export function transpose(a: Matrix): Matrix {
  const rows = a.length
  const cols = a[0].length
  const out = zeros(cols, rows)
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      out[j][i] = a[i][j]
    }
  }
  return out
}

export function addBias(a: Matrix, bias: number[]): Matrix {
  return a.map((row) => row.map((v, j) => v + bias[j]))
}

export function elementwise(a: Matrix, b: Matrix, fn: (x: number, y: number) => number): Matrix {
  return a.map((row, i) => row.map((v, j) => fn(v, b[i][j])))
}

export function map(a: Matrix, fn: (x: number) => number): Matrix {
  return a.map((row) => row.map(fn))
}

export function scale(a: Matrix, s: number): Matrix {
  return a.map((row) => row.map((v) => v * s))
}

// Sum of each column across all rows, used to reduce a batch's per-sample
// gradients down to a single bias gradient.
export function sumColumns(a: Matrix): number[] {
  const cols = a[0].length
  const out = new Array(cols).fill(0)
  for (const row of a) {
    for (let j = 0; j < cols; j++) out[j] += row[j]
  }
  return out
}

// Row-wise softmax: each row's entries become a probability distribution
// over that row's classes. The max is subtracted first purely for numeric
// stability (exp of a large raw score overflows before it gets normalized),
// it does not change the result since it cancels out of the ratio.
export function softmaxRows(z: Matrix): Matrix {
  return z.map((row) => {
    const max = Math.max(...row)
    const exps = row.map((v) => Math.exp(v - max))
    const sum = exps.reduce((a, b) => a + b, 0)
    return exps.map((v) => v / sum)
  })
}
