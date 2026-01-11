import {
	isLowerLevelRoutingFilter,
	isUpperLevelRoutingFilter,
	isCurrentLevelRoutingFilter,
	leadsToLowerLevelRoutingFilter,
	leadsUpToCurrentLevelRoutingFilter,
	leadsDownToCurrentLevelRoutingFilter,
	leadsToUpperLevelRoutingFilter
} from './layerFilters';

/// Routing path current level line color.
const routingPathFillColor = '#42a5f5';
/// Routing path current level line outline color.
const routingPathOutlineColor = '#0077c2';
/// Routing path other level line color.
const routingPathOtherLevelFillColor = '#aaaaaa';
/// Routing path other level line outline color.
const routingPathOtherLevelOutlineColor = '#555555';
/// Routing path line color.
const routingPathWidth = 7;
/// Routing path line color.
const routingPathOutlineWidth = routingPathWidth + 2;

export const layers = [
	// Indoor routing - Outline - Current level \\

	{
		id: 'indoor-routing-path-current-outline',
		type: 'line',
		filter: isCurrentLevelRoutingFilter,
		layout: {
			'line-join': 'round',
			'line-cap': 'round'
		},
		paint: {
			'line-color': routingPathOutlineColor,
			'line-width': routingPathOutlineWidth
		}
	},

	// Indoor routing - Lower level connecting path segments \\

	{
		id: 'indoor-routing-lower-path-down-outline',
		type: 'line',
		filter: leadsToLowerLevelRoutingFilter,
		layout: {
			'line-join': 'round'
		},
		paint: {
			'line-width': routingPathOutlineWidth,
			'line-gradient': [
				'interpolate',
				['linear'],
				['line-progress'],
				0,
				routingPathOutlineColor,
				1,
				routingPathOtherLevelOutlineColor
			]
		}
	},
	{
		id: 'indoor-routing-lower-path-down',
		type: 'line',
		filter: leadsToLowerLevelRoutingFilter,
		layout: {
			'line-join': 'round'
		},
		paint: {
			'line-width': routingPathWidth,
			'line-gradient': [
				'interpolate',
				['linear'],
				['line-progress'],
				0,
				routingPathFillColor,
				1,
				routingPathOtherLevelFillColor
			]
		}
	},
	{
		id: 'indoor-routing-lower-path-up-outline',
		type: 'line',
		filter: leadsUpToCurrentLevelRoutingFilter,
		layout: {
			'line-join': 'round'
		},
		paint: {
			'line-width': routingPathOutlineWidth,
			'line-gradient': [
				'interpolate',
				['linear'],
				['line-progress'],
				0,
				routingPathOtherLevelOutlineColor,
				1,
				routingPathOutlineColor
			]
		}
	},
	{
		id: 'indoor-routing-lower-path-up',
		type: 'line',
		filter: leadsUpToCurrentLevelRoutingFilter,
		layout: {
			'line-join': 'round'
		},
		paint: {
			'line-width': routingPathWidth,
			'line-gradient': [
				'interpolate',
				['linear'],
				['line-progress'],
				0,
				routingPathOtherLevelFillColor,
				1,
				routingPathFillColor
			]
		}
	},

	// Indoor routing - Outline - Upper level connecting path segments \\

	{
		id: 'indoor-routing-upper-path-down-outline',
		type: 'line',
		filter: leadsDownToCurrentLevelRoutingFilter,
		layout: {
			'line-join': 'round'
		},
		paint: {
			'line-width': routingPathOutlineWidth,
			// 'line-gradient' must be specified using an expression
			// with the special 'line-progress' property
			// the source must have the 'lineMetrics' option set to true
			// note the line points have to be ordered so it fits (the direction of the line)
			// because no other expression are supported here
			'line-gradient': [
				'interpolate',
				['linear'],
				['line-progress'],
				0,
				routingPathOtherLevelOutlineColor,
				1,
				routingPathOutlineColor
			]
		}
	},
	{
		id: 'indoor-routing-upper-path-up-outline',
		type: 'line',
		filter: leadsToUpperLevelRoutingFilter,
		layout: {
			'line-join': 'round'
		},
		paint: {
			'line-width': routingPathOutlineWidth,
			'line-gradient': [
				'interpolate',
				['linear'],
				['line-progress'],
				0,
				routingPathOutlineColor,
				1,
				routingPathOtherLevelOutlineColor
			]
		}
	},

	// Indoor routing - Concealed edges outline - Below current level \\

	{
		id: 'indoor-routing-path-concealed-below-outline',
		// required, otherwise line-dasharray will scale with metrics
		type: 'line',
		filter: isLowerLevelRoutingFilter,
		layout: {
			'line-join': 'round'
		},
		paint: {
			'line-color': routingPathOtherLevelOutlineColor,
			'line-width': 2,
			'line-gap-width': 6,
			'line-dasharray': ['literal', [2, 2]]
		}
	},

	// Indoor routing - Outline - Above current level \\

	{
		id: 'indoor-routing-path-above-outline',
		type: 'line',
		filter: isUpperLevelRoutingFilter,
		layout: {
			'line-join': 'round',
			'line-cap': 'round'
		},
		paint: {
			'line-color': routingPathOtherLevelOutlineColor,
			'line-width': routingPathOutlineWidth
		}
	},

	// Indoor routing - Fill - Current level \\

	{
		id: 'indoor-routing-path-current',
		type: 'line',
		filter: isCurrentLevelRoutingFilter,
		layout: {
			'line-join': 'round',
			'line-cap': 'round'
		},
		paint: {
			'line-color': routingPathFillColor,
			'line-width': routingPathWidth
		}
	},

	// Indoor routing - Fill - Upper level connecting path segments \\

	{
		id: 'indoor-routing-upper-path-down',
		type: 'line',
		filter: leadsDownToCurrentLevelRoutingFilter,
		layout: {
			'line-join': 'round',
			'line-cap': 'round'
		},
		paint: {
			'line-width': routingPathWidth,
			'line-gradient': [
				'interpolate',
				['linear'],
				['line-progress'],
				0,
				routingPathOtherLevelFillColor,
				1,
				routingPathFillColor
			]
		}
	},
	{
		id: 'indoor-routing-upper-path-up',
		type: 'line',
		filter: leadsToUpperLevelRoutingFilter,
		layout: {
			'line-join': 'round',
			'line-cap': 'round'
		},
		paint: {
			'line-width': routingPathWidth,
			'line-gradient': [
				'interpolate',
				['linear'],
				['line-progress'],
				0,
				routingPathFillColor,
				1,
				routingPathOtherLevelFillColor
			]
		}
	},

	// Indoor routing - Fill - Above current level \\

	{
		id: 'indoor-routing-path-above',
		type: 'line',
		filter: isUpperLevelRoutingFilter,
		layout: {
			'line-join': 'round',
			'line-cap': 'round'
		},
		paint: {
			'line-color': routingPathOtherLevelFillColor,
			'line-width': routingPathWidth
		}
	}
] as const;
