// Provisions the map font glyphs into static/glyphs/ at build time.
//
// The glyphs (Noto Sans Regular/Bold SDF PBFs, ~68 MB) are intentionally NOT
// committed (see .gitignore). They are sourced from the pbf-sdf-fonts package
// that the `tiles` dependency already pulls in via .pkg, so in a normal build
// there is no network access: we just copy from deps/pbf-sdf-fonts/res.
// If that dependency is not present (e.g. a stand-alone UI checkout), we fall
// back to downloading the same first-party repo from GitHub.
//
// Idempotent: if the glyphs are already present, this is a no-op.

import { existsSync, mkdirSync, cpSync, writeFileSync, rmSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const uiDir = join(dirname(fileURLToPath(import.meta.url)), '..');
const glyphsDir = join(uiDir, 'static', 'glyphs');
const depRes = join(uiDir, '..', 'deps', 'pbf-sdf-fonts', 'res');
const FONT_REPO_TARBALL =
	'https://github.com/motis-project/pbf-sdf-fonts/archive/refs/heads/master.tar.gz';

const log = (...a) => console.log('[map-assets]', ...a);

if (existsSync(join(glyphsDir, 'Noto Sans Regular', '0-255.pbf'))) {
	log('glyphs already present, skipping');
	process.exit(0);
}

mkdirSync(glyphsDir, { recursive: true });

if (existsSync(join(depRes, 'Noto Sans Regular'))) {
	log('copying glyphs from deps/pbf-sdf-fonts/res (.pkg dependency)');
	cpSync(depRes, glyphsDir, { recursive: true });
} else {
	log('deps/pbf-sdf-fonts not found — downloading from motis-project/pbf-sdf-fonts');
	const tgz = join(glyphsDir, '.pbf-sdf-fonts.tmp.tgz');
	const res = await fetch(FONT_REPO_TARBALL);
	if (!res.ok) throw new Error(`download failed: HTTP ${res.status}`);
	writeFileSync(tgz, Buffer.from(await res.arrayBuffer()));
	// extract only <tarball>/res/* directly into static/glyphs/
	const r = spawnSync(
		'tar',
		['-xzf', tgz, '-C', glyphsDir, '--strip-components=2', 'pbf-sdf-fonts-master/res'],
		{ stdio: 'inherit' }
	);
	rmSync(tgz, { force: true });
	if (r.status !== 0) throw new Error('tar extraction failed');
}

log('glyphs ready in static/glyphs/');
