// Fetch the networks the app offers (models.json) from zephyr's GitHub releases into
// public/models/<name>/ (encoder.onnx, temporal.onnx, meta.json, as tools/export_onnx.py
// writes them) and list them in public/models/index.json. Files already there are kept
// unless --force. Run before `npm run build` or `npm run dev`: `npm run models`.
import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const REPO = "https://github.com/AllenNeuralDynamics/zephyr";
const FILES = ["encoder.onnx", "temporal.onnx", "meta.json"];

const web = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const out = resolve(web, "public/models");
const force = process.argv.includes("--force");
const networks = JSON.parse(await readFile(resolve(web, "models.json"), "utf8"));

const exists = (path) => stat(path).then(() => true, () => false);

for (const { name, release } of networks) {
  await mkdir(resolve(out, name), { recursive: true });
  for (const file of FILES) {
    const path = resolve(out, name, file);
    if (!force && (await exists(path))) continue;
    const url = `${REPO}/releases/download/${release}/${file}`;
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: ${response.status} ${response.statusText}`);
    await writeFile(path, Buffer.from(await response.arrayBuffer()));
    console.log(`${name}/${file} <- ${url}`);
  }
}
await writeFile(resolve(out, "index.json"), JSON.stringify(networks.map((n) => n.name), null, 2));
console.log(`models: ${networks.map((n) => n.name).join(", ")}`);
