import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vitest/config';
export default defineConfig({
	plugins: [sveltekit()],
	test: {
		include: ['src/**/*.{test,spec}.{js,ts}']
	},
	server: {
		fs: { strict: false }
	},
	build: {
		sourcemap: true,
		rollupOptions: {
			output: {
				manualChunks: (id) => {
					if (id.includes('node_modules')) {
						if (id.includes('deck.gl')) return 'deck-vendor';
						if (id.includes('svelte')) return 'svelte-vendor';
						if (id.includes('luma.gl')) return 'luma-vendor';
						if (id.includes('.gl')) return 'gl-vendor';
						return 'other-vendor';
					}
				}
			}
		}
	}
});
