// Render built-in Shapes items with the JS port and print them as JSON for the Python parity test.
// Usage: node shapes_dump.mjs <seed> <size> <index> [<index> ...]
import { render } from '../../site/js/shapes.js';

const [seed, size, ...idx] = process.argv.slice(2).map(Number);
const out = idx.map((i) => {
  const { img, label } = render(seed, i, size);
  return { index: i, label, img: Array.from(img) };
});
process.stdout.write(JSON.stringify(out));
