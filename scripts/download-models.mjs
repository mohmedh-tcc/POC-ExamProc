import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

const wasmDir = path.join(rootDir, 'public', 'wasm');
const modelsDir = path.join(rootDir, 'public', 'models');

fs.mkdirSync(wasmDir, { recursive: true });
fs.mkdirSync(modelsDir, { recursive: true });

console.log('Copying MediaPipe tasks-vision WASM files...');
const mpWasmSource = path.join(rootDir, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm');
if (fs.existsSync(mpWasmSource)) {
  for (const file of fs.readdirSync(mpWasmSource)) {
    const dest = path.join(wasmDir, file);
    fs.copyFileSync(path.join(mpWasmSource, file), dest);
    if (file === 'vision_wasm_internal.js') {
      let content = fs.readFileSync(dest, 'utf8');
      if (!content.includes('globalThis.ModuleFactory = ModuleFactory')) {
        content += '\nif (typeof globalThis !== "undefined") { globalThis.ModuleFactory = ModuleFactory; }\nif (typeof self !== "undefined") { self.ModuleFactory = ModuleFactory; }\n';
        fs.writeFileSync(dest, content);
      }
    }
    console.log(`  ✓ Copied ${file}`);
  }
} else {
  console.error(`MediaPipe WASM directory not found: ${mpWasmSource}`);
}

console.log('\nCopying Silero VAD assets...');
const vadDist = path.join(rootDir, 'node_modules', '@ricky0123', 'vad-web', 'dist');
if (fs.existsSync(vadDist)) {
  const vadFiles = ['silero_vad_v5.onnx', 'silero_vad_legacy.onnx', 'vad.worklet.bundle.min.js'];
  for (const file of vadFiles) {
    const src = path.join(vadDist, file);
    if (fs.existsSync(src)) {
      fs.copyFileSync(src, path.join(wasmDir, file));
      console.log(`  ✓ Copied ${file}`);
    }
  }
}

console.log('\nCopying onnxruntime-web WASM files...');
const ortDist = path.join(rootDir, 'node_modules', 'onnxruntime-web', 'dist');
if (fs.existsSync(ortDist)) {
  const ortFiles = [
    'ort-wasm-simd-threaded.wasm',
    'ort-wasm-simd.wasm',
    'ort-wasm.wasm',
    'ort-wasm-threaded.wasm'
  ];
  for (const file of ortFiles) {
    const src = path.join(ortDist, file);
    if (fs.existsSync(src)) {
      fs.copyFileSync(src, path.join(wasmDir, file));
      console.log(`  ✓ Copied ${file}`);
    }
  }
}

// Download model assets if not present
const models = [
  {
    name: 'face_landmarker.task',
    url: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
    expectedMinSize: 3_000_000,
  },
  {
    name: 'efficientdet_lite0.tflite',
    url: 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite',
    expectedMinSize: 4_000_000,
  }
];

async function downloadFile(url, destPath) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to download ${url}: ${response.status} ${response.statusText}`);
  }
  const buffer = await response.arrayBuffer();
  fs.writeFileSync(destPath, Buffer.from(buffer));
}

console.log('\nChecking model assets...');
for (const model of models) {
  const destPath = path.join(modelsDir, model.name);
  if (fs.existsSync(destPath)) {
    const stat = fs.statSync(destPath);
    if (stat.size >= model.expectedMinSize) {
      console.log(`  ✓ ${model.name} exists (${(stat.size / 1024 / 1024).toFixed(2)} MB)`);
      continue;
    }
    console.log(`  ! ${model.name} size mismatch, re-downloading...`);
  }
  console.log(`  ⬇ Downloading ${model.name} from ${model.url}...`);
  await downloadFile(model.url, destPath);
  const stat = fs.statSync(destPath);
  console.log(`  ✓ Downloaded ${model.name} (${(stat.size / 1024 / 1024).toFixed(2)} MB)`);
}

console.log('\nAll offline assets verified successfully!');
