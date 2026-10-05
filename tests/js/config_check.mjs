// Validate every JSON file given on the command line with the web validator.
// Prints one JSON line: {"<file>": ["error", ...], ...}. The Python side compares
// accept/reject (and the exact messages) with jepa_studio.config.validate.
import { readFileSync } from 'node:fs';
import { validate, parseConfigText } from '../../site/js/config.js';

const schema = JSON.parse(readFileSync(new URL('../../site/schema/config.schema.json', import.meta.url), 'utf8'));
const out = {};
for (const f of process.argv.slice(2)) {
  out[f] = validate(parseConfigText(readFileSync(f, 'utf8')), schema);
}
process.stdout.write(JSON.stringify(out));
