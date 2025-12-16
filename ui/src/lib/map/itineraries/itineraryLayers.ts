import {
	_isLowerLevelRoutingFilter,
	_isUpperLevelRoutingFilter,
	_isCurrentLevelRoutingFilter,
	_leadsToLowerLevelRoutingFilter,
	_leadsUpToCurrentLevelRoutingFilter,
	_leadsDownToCurrentLevelRoutingFilter,
	_leadsToUpperLevelRoutingFilter,
	_connectsToCurrentLevelRoutingFilter,
	_isCurrentLevelFilter,
	_ceilFromLevel,
	_ceilToLevel,
	_floorFromLevel,
	_floorToLevel,
	_isLowerLevelFilter
} from './layerFilters';

/// Routing path current level line color.
const _routingPathFillColor = '#42a5f5';
/// Routing path current level line outline color.
const _routingPathOutlineColor = '#0077c2';
/// Routing path other level line color.
const _routingPathOtherLevelFillColor = '#aaaaaa';
/// Routing path other level line outline color.
const _routingPathOtherLevelOutlineColor = '#555555';
/// Routing path line color.
const _routingPathWidth = 7;
/// Routing path line color.
const _routingPathOutlineWidth = _routingPathWidth + 2;

export const layers = [
	// Indoor routing - Outline - Current level \\

	{
		id: 'indoor-routing-path-current-outline',
		type: 'line',
		filter: _isCurrentLevelRoutingFilter,
		layout: {
			'line-join': 'round',
			'line-cap': 'round'
		},
		paint: {
			'line-color': _routingPathOutlineColor,
			'line-width': _routingPathOutlineWidth
		}
	},

	// Indoor routing - Lower level connecting path segments \\

	{
		id: 'indoor-routing-lower-path-down-outline',
		type: 'line',
		filter: _leadsToLowerLevelRoutingFilter,
		layout: {
			'line-join': 'round'
		},
		paint: {
			'line-width': _routingPathOutlineWidth,
			'line-gradient': [
				'interpolate',
				['linear'],
				['line-progress'],
				0,
				_routingPathOutlineColor,
				1,
				_routingPathOtherLevelOutlineColor
			]
		}
	},
	{
		id: 'indoor-routing-lower-path-down',
		type: 'line',
		filter: _leadsToLowerLevelRoutingFilter,
		layout: {
			'line-join': 'round'
		},
		paint: {
			'line-width': _routingPathWidth,
			'line-gradient': [
				'interpolate',
				['linear'],
				['line-progress'],
				0,
				_routingPathFillColor,
				1,
				_routingPathOtherLevelFillColor
			]
		}
	},
	{
		id: 'indoor-routing-lower-path-up-outline',
		type: 'line',
		filter: _leadsUpToCurrentLevelRoutingFilter,
		layout: {
			'line-join': 'round'
		},
		paint: {
			'line-width': _routingPathOutlineWidth,
			'line-gradient': [
				'interpolate',
				['linear'],
				['line-progress'],
				0,
				_routingPathOtherLevelOutlineColor,
				1,
				_routingPathOutlineColor
			]
		}
	},
	{
		id: 'indoor-routing-lower-path-up',
		type: 'line',
		filter: _leadsUpToCurrentLevelRoutingFilter,
		layout: {
			'line-join': 'round'
		},
		paint: {
			'line-width': _routingPathWidth,
			'line-gradient': [
				'interpolate',
				['linear'],
				['line-progress'],
				0,
				_routingPathOtherLevelFillColor,
				1,
				_routingPathFillColor
			]
		}
	},

	// Indoor routing - Outline - Upper level connecting path segments \\

	{
		id: 'indoor-routing-upper-path-down-outline',
		type: 'line',
		filter: _leadsDownToCurrentLevelRoutingFilter,
		layout: {
			'line-join': 'round'
		},
		paint: {
			'line-width': _routingPathOutlineWidth,
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
				_routingPathOtherLevelOutlineColor,
				1,
				_routingPathOutlineColor
			]
		}
	},
	{
		id: 'indoor-routing-upper-path-up-outline',
		type: 'line',
		filter: _leadsToUpperLevelRoutingFilter,
		layout: {
			'line-join': 'round'
		},
		paint: {
			'line-width': _routingPathOutlineWidth,
			'line-gradient': [
				'interpolate',
				['linear'],
				['line-progress'],
				0,
				_routingPathOutlineColor,
				1,
				_routingPathOtherLevelOutlineColor
			]
		}
	},

	// Indoor routing - Concealed edges outline - Below current level \\

	{
		id: 'indoor-routing-path-concealed-below-outline',
		// required, otherwise line-dasharray will scale with metrics
		type: 'line',
		filter: _isLowerLevelRoutingFilter,
		layout: {
			'line-join': 'round'
		},
		paint: {
			'line-color': _routingPathOtherLevelOutlineColor,
			'line-width': 2,
			'line-gap-width': 6,
			'line-dasharray': ['literal', [2, 2]]
		}
	},

	// Indoor routing - Outline - Above current level \\

	{
		id: 'indoor-routing-path-above-outline',
		type: 'line',
		filter: _isUpperLevelRoutingFilter,
		layout: {
			'line-join': 'round',
			'line-cap': 'round'
		},
		paint: {
			'line-color': _routingPathOtherLevelOutlineColor,
			'line-width': _routingPathOutlineWidth
		}
	},

	// Indoor routing - Fill - Current level \\

	{
		id: 'indoor-routing-path-current',
		type: 'line',
		filter: _isCurrentLevelRoutingFilter,
		layout: {
			'line-join': 'round',
			'line-cap': 'round'
		},
		paint: {
			'line-color': _routingPathFillColor,
			'line-width': _routingPathWidth
		}
	},

	// Indoor routing - Fill - Upper level connecting path segments \\

	{
		id: 'indoor-routing-upper-path-down',
		type: 'line',
		filter: _leadsDownToCurrentLevelRoutingFilter,
		layout: {
			'line-join': 'round',
			'line-cap': 'round'
		},
		paint: {
			'line-width': _routingPathWidth,
			'line-gradient': [
				'interpolate',
				['linear'],
				['line-progress'],
				0,
				_routingPathOtherLevelFillColor,
				1,
				_routingPathFillColor
			]
		}
	},
	{
		id: 'indoor-routing-upper-path-up',
		type: 'line',
		filter: _leadsToUpperLevelRoutingFilter,
		layout: {
			'line-join': 'round',
			'line-cap': 'round'
		},
		paint: {
			'line-width': _routingPathWidth,
			'line-gradient': [
				'interpolate',
				['linear'],
				['line-progress'],
				0,
				_routingPathFillColor,
				1,
				_routingPathOtherLevelFillColor
			]
		}
	},

	// Indoor routing - Fill - Above current level \\

	{
		id: 'indoor-routing-path-above',
		type: 'line',
		filter: _isUpperLevelRoutingFilter,
		layout: {
			'line-join': 'round',
			'line-cap': 'round'
		},
		paint: {
			'line-color': _routingPathOtherLevelFillColor,
			'line-width': _routingPathWidth
		}
	}
] as const;
