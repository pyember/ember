import React, {useEffect, useState} from 'react';
import {Box, Text} from 'ink';

type Point3D = [number, number, number];
type Projected = [number, number, number];

const BASE_WIDTH = 32;
const BASE_HEIGHT = 18;
// Extra width headroom to prevent frame wrapping.
const MAX_WIDE_GLYPHS = 4;
export const CUBE_VIEWPORT_WIDTH = BASE_WIDTH + MAX_WIDE_GLYPHS;
const CUBE_COLOR = 'blue';
const FPS = 24;

// Unified gradient for a cleaner, more solid look
// Start with a visible char instead of space to avoid invisible faces
const GRADIENT = '.:-=+*#%@';

const FOCAL = 2.6;
const CAMERA_Z = 4.0;
// Lively rotation speed — engaging motion.
const ROTATE_SPEEDS: [number, number, number] = [0.7, 0.95, 0.45];
const FACE_STEPS = 16; // Increased density for smoother faces
const MASK_FACE_INDEX = 1;
// Horizontal band spanning most of the face width, thin vertical slice.
const MASK_U0 = 0.2;
const MASK_U1 = 0.8;
const MASK_V0 = 0.4;
const MASK_V1 = 0.6;
const MASK_CHAR = '↗';  // Northeast arrow for brand mark

const CUBE_VERTICES: Point3D[] = [
  [-1, -1, -1],
  [1, -1, -1],
  [-1, 1, -1],
  [1, 1, -1],
  [-1, -1, 1],
  [1, -1, 1],
  [-1, 1, 1],
  [1, 1, 1],
];

const FACES: Array<[number, number, number, number]> = [
  [0, 1, 3, 2], // back (-z)
  [4, 5, 7, 6], // front (+z)
  [0, 1, 5, 4], // bottom (-y)
  [2, 3, 7, 6], // top (+y)
  [0, 2, 6, 4], // left (-x)
  [1, 3, 7, 5], // right (+x)
];

const EDGES: Array<[number, number]> = [
  [0, 1],
  [1, 3],
  [3, 2],
  [2, 0],
  [4, 5],
  [5, 7],
  [7, 6],
  [6, 4],
  [0, 4],
  [1, 5],
  [2, 6],
  [3, 7],
];

const isWideChar = (char: string): boolean => /[^\u0000-\u00ff]/.test(char);

const measureWidth = (value: string): number =>
  Array.from(value).reduce((width, char) => width + (isWideChar(char) ? 2 : 1), 0);

const padToViewport = (line: string): string => {
  const displayWidth = measureWidth(line);
  if (displayWidth >= CUBE_VIEWPORT_WIDTH) return line;

  const totalPadding = CUBE_VIEWPORT_WIDTH - displayWidth;
  const left = Math.floor(totalPadding / 2);
  const right = totalPadding - left;
  return `${' '.repeat(left)}${line}${' '.repeat(right)}`;
};

const normalizeFrame = (lines: string[]): string[] => lines.map(padToViewport);

export const CubeAnimation: React.FC = () => {
  const [frame, setFrame] = useState<string[]>(() =>
    normalizeFrame(renderFrame(0, BASE_WIDTH, BASE_HEIGHT, FOCAL))
  );

  useEffect(() => {
    let cancelled = false;
    let t = 0;
    const dt = 1 / FPS;
    const id = setInterval(() => {
      if (cancelled) return;
      t += dt;
      setFrame(normalizeFrame(renderFrame(t, BASE_WIDTH, BASE_HEIGHT, FOCAL)));
    }, 1000 / FPS);

    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  return (
    <Box flexDirection="column" alignItems="center" width={CUBE_VIEWPORT_WIDTH}>
      {frame.map((line, idx) => (
        <Text key={idx} color={CUBE_COLOR} wrap="truncate">{line}</Text>
      ))}
    </Box>
  );
};

function rotatePoint([x, y, z]: Point3D, ax: number, ay: number, az: number): Point3D {
  const cx = Math.cos(ax);
  const sx = Math.sin(ax);
  const cy = Math.cos(ay);
  const sy = Math.sin(ay);
  const cz = Math.cos(az);
  const sz = Math.sin(az);

  let y1 = y * cx - z * sx;
  let z1 = y * sx + z * cx;
  let x1 = x * cy + z1 * sy;
  let z2 = -x * sy + z1 * cy;
  let x2 = x1 * cz - y1 * sz;
  let y2 = x1 * sz + y1 * cz;

  return [x2, y2, z2];
}

function project(point: Point3D, width: number, height: number, focal: number): Projected {
  const [x, y, z] = point;
  const zCam = z + CAMERA_Z;
  const x2d = Math.floor(width / 2 + (x * focal) / zCam * (width / 3));
  const y2d = Math.floor(height / 2 - (y * focal) / zCam * (height / 3));
  return [x2d, y2d, zCam];
}

function faceNormal(vertices: Point3D[]): Point3D {
  const [a, b, c] = vertices;
  const e1: Point3D = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const e2: Point3D = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  const nx = e1[1] * e2[2] - e1[2] * e2[1];
  const ny = e1[2] * e2[0] - e1[0] * e2[2];
  const nz = e1[0] * e2[1] - e1[1] * e2[0];
  const length = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
  return [nx / length, ny / length, nz / length];
}

function drawEdge(
  start: Projected,
  end: Projected,
  buffer: string[][],
  depth: number[][]
): void {
  const [x0, y0, z0] = start;
  const [x1, y1, z1] = end;
  const steps = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0), 1);
  for (let i = 0; i <= steps; i += 1) {
    const t = steps ? i / steps : 0;
    const x = Math.round(x0 + (x1 - x0) * t);
    const y = Math.round(y0 + (y1 - y0) * t);
    const z = z0 + (z1 - z0) * t;
    if (y < 0 || y >= buffer.length || x < 0 || x >= buffer[0].length) continue;
    // Small depth bias to ensure edges sit "on top" of faces
    if (z < depth[y][x] - 0.05) {
      depth[y][x] = z;
      buffer[y][x] = '.';
    }
  }
}

function maskChar(u: number, v: number): string | null {
  // Return arrow character when within the stripe region.
  if (MASK_U0 <= u && u <= MASK_U1 && MASK_V0 <= v && v <= MASK_V1) {
    return MASK_CHAR;
  }
  return null;
}

function rotationSpeedScalar(t: number): number {
  // Subtle speed variation for smooth, organic motion.
  const wave = Math.sin(t * 0.3);
  return 0.75 + 0.05 * wave;
}

function renderFrame(t: number, width: number, height: number, focal: number): string[] {
  const scalar = rotationSpeedScalar(t);
  const ax = t * ROTATE_SPEEDS[0] * scalar;
  const ay = t * ROTATE_SPEEDS[1] * scalar;
  const az = t * ROTATE_SPEEDS[2] * scalar;
  const rotated = CUBE_VERTICES.map((p) => rotatePoint(p, ax, ay, az));
  const projected = rotated.map((p) => project(p, width, height, focal));

  const normals = FACES.map((face) =>
    faceNormal(face.map((i) => rotated[i]))
  );

  const buffer = Array.from({length: height}, () => Array(width).fill(' '));
  const depth = Array.from({length: height}, () => Array(width).fill(Number.POSITIVE_INFINITY));

  for (let fi = 0; fi < FACES.length; fi += 1) {
    const face = FACES[fi];
    const normal = normals[fi];
    // Lighting based on normal Z (facing camera)
    // Added ambient light (0.2) so no face is completely invisible
    const baseBrightness = Math.max(0.2, -normal[2]);
    
    // Backface culling: strict culling if normal Z is positive (facing away)
    if (normal[2] > 0) continue;
    
    const [i0, i1, i2, i3] = face.map((idx) => rotated[idx]);

    for (let iu = 0; iu <= FACE_STEPS; iu += 1) {
      const u = iu / FACE_STEPS;
      for (let iv = 0; iv <= FACE_STEPS; iv += 1) {
        const v = iv / FACE_STEPS;
        const x =
          i0[0] * (1 - u) * (1 - v) +
          i1[0] * u * (1 - v) +
          i2[0] * u * v +
          i3[0] * (1 - u) * v;
        const y =
          i0[1] * (1 - u) * (1 - v) +
          i1[1] * u * (1 - v) +
          i2[1] * u * v +
          i3[1] * (1 - u) * v;
        const z =
          i0[2] * (1 - u) * (1 - v) +
          i1[2] * u * (1 - v) +
          i2[2] * u * v +
          i3[2] * (1 - u) * v;

        const [x2d, y2d, zCam] = project([x, y, z], width, height, focal);
        if (x2d < 0 || x2d >= width || y2d < 0 || y2d >= height) continue;
        if (zCam >= depth[y2d][x2d]) continue;

        // Map brightness to gradient
        const idx = Math.floor(baseBrightness * (GRADIENT.length - 1));
        const clampedIdx = Math.max(0, Math.min(GRADIENT.length - 1, idx));
        let ch = GRADIENT[clampedIdx];

        if (fi === MASK_FACE_INDEX) {
          const masked = maskChar(u, v);
          if (masked) ch = masked;
        }

        depth[y2d][x2d] = zCam;
        buffer[y2d][x2d] = ch;
      }
    }
  }

  EDGES.forEach((edge) => {
    drawEdge(projected[edge[0]], projected[edge[1]], buffer, depth);
  });

  return buffer.map((row) => row.join(''));
}
