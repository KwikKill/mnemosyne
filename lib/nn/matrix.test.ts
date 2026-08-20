import { describe, expect, it } from "vitest"
import { addBias, elementwise, map, multiply, scale, softmaxRows, sumColumns, transpose, zeros } from "./matrix"

describe("zeros", () => {
  it("fills every cell with 0 at the requested shape", () => {
    expect(zeros(2, 3)).toEqual([
      [0, 0, 0],
      [0, 0, 0],
    ])
  })
})

describe("multiply", () => {
  it("matches hand-computed matrix products", () => {
    const a = [
      [1, 2],
      [3, 4],
    ]
    const b = [
      [5, 6],
      [7, 8],
    ]
    // [1*5+2*7, 1*6+2*8] = [19, 22]; [3*5+4*7, 3*6+4*8] = [43, 50]
    expect(multiply(a, b)).toEqual([
      [19, 22],
      [43, 50],
    ])
  })

  it("supports non-square shapes (batch x features times features x units)", () => {
    const batch = [
      [1, 0, 2],
      [0, 1, 1],
    ] // 2x3
    const weights = [[1], [2], [3]] // 3x1
    expect(multiply(batch, weights)).toEqual([[7], [5]])
  })
})

describe("transpose", () => {
  it("flips rows and columns", () => {
    const a = [
      [1, 2, 3],
      [4, 5, 6],
    ]
    expect(transpose(a)).toEqual([
      [1, 4],
      [2, 5],
      [3, 6],
    ])
  })

  it("is its own inverse", () => {
    const a = [
      [1, 2],
      [3, 4],
      [5, 6],
    ]
    expect(transpose(transpose(a))).toEqual(a)
  })
})

describe("addBias", () => {
  it("adds the same bias vector to every row", () => {
    const a = [
      [1, 2],
      [3, 4],
    ]
    expect(addBias(a, [10, 100])).toEqual([
      [11, 102],
      [13, 104],
    ])
  })
})

describe("elementwise / map / scale", () => {
  it("elementwise applies a binary function position by position", () => {
    const a = [[1, 2]]
    const b = [[10, 20]]
    expect(elementwise(a, b, (x, y) => x + y)).toEqual([[11, 22]])
  })

  it("map applies a unary function to every entry", () => {
    expect(map([[1, -2, 3]], (x) => x * x)).toEqual([[1, 4, 9]])
  })

  it("scale multiplies every entry by a scalar", () => {
    expect(scale([[1, -2, 3]], 2)).toEqual([[2, -4, 6]])
  })
})

describe("sumColumns", () => {
  it("sums each column down all rows", () => {
    const a = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]
    expect(sumColumns(a)).toEqual([12, 15, 18])
  })
})

describe("softmaxRows", () => {
  it("produces a probability distribution per row (sums to 1, all positive)", () => {
    const rows = softmaxRows([
      [1, 2, 3],
      [-5, 0, 5],
    ])
    for (const row of rows) {
      const sum = row.reduce((a, b) => a + b, 0)
      expect(sum).toBeCloseTo(1, 10)
      for (const v of row) expect(v).toBeGreaterThan(0)
    }
  })

  it("assigns the highest probability to the largest raw score", () => {
    const [row] = softmaxRows([[1, 5, 2]])
    expect(row[1]).toBeGreaterThan(row[0])
    expect(row[1]).toBeGreaterThan(row[2])
  })

  it("stays numerically stable for large inputs that would overflow exp() unshifted", () => {
    const [row] = softmaxRows([[1000, 1001, 1002]])
    expect(row.every((v) => Number.isFinite(v))).toBe(true)
    const sum = row.reduce((a, b) => a + b, 0)
    expect(sum).toBeCloseTo(1, 10)
  })

  it("is uniform when every logit in a row is equal", () => {
    const [row] = softmaxRows([[3, 3, 3, 3]])
    for (const v of row) expect(v).toBeCloseTo(0.25, 10)
  })
})
