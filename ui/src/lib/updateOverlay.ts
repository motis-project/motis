import { IconLayer } from '@deck.gl/layers';
import { MapboxOverlay } from '@deck.gl/mapbox';

export const updateOverlayLayers = (l: IconLayer, layers: IconLayer[], overlay: MapboxOverlay) => {
	const idx = layers.findIndex(layer => layer.id === l.id);
	if (idx !== -1) {
		layers[idx] = l;
	} else {
		layers.push(l);
	}
	overlay.setProps({ layers: [...layers] });
};
