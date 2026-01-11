// routing layer \\

import type { ExpressionFilterSpecification } from 'maplibre-gl';

export const currentLevel: ExpressionFilterSpecification = ['coalesce', ['get', 'level'], 0];
export const ceilFromLevel: ExpressionFilterSpecification = [
	'coalesce',
	['ceil', ['to-number', ['get', 'fromLevel']]],
	0
];
export const ceilToLevel: ExpressionFilterSpecification = [
	'coalesce',
	['ceil', ['to-number', ['get', 'toLevel']]],
	0
];
export const floorFromLevel: ExpressionFilterSpecification = [
	'coalesce',
	['floor', ['to-number', ['get', 'fromLevel']]],
	0
];
export const floorToLevel: ExpressionFilterSpecification = [
	'coalesce',
	['floor', ['to-number', ['get', 'toLevel']]],
	0
];

/// Filter to match all connections that lie on, cross or connect to the current level.

export const connectsToCurrentLevelRoutingFilter: ExpressionFilterSpecification = [
	'all',
	['<=', ['min', floorToLevel, floorFromLevel], currentLevel],
	['>=', ['max', ceilToLevel, ceilFromLevel], currentLevel]
];

/// Filter to match path connections on the current level.

export const isCurrentLevelRoutingFilter: ExpressionFilterSpecification = [
	'any',
	['all', ['==', ceilFromLevel, currentLevel], ['==', ceilToLevel, currentLevel]],
	['all', ['==', floorFromLevel, currentLevel], ['==', floorToLevel, currentLevel]]
];

/// Filter to match path connections on any lower level that do not connect to current level.

export const isLowerLevelRoutingFilter: ExpressionFilterSpecification = [
	'any',
	['all', ['<', ceilFromLevel, currentLevel], ['<', ceilToLevel, currentLevel]],
	['all', ['<', floorFromLevel, currentLevel], ['<', floorToLevel, currentLevel]]
];

/// Filter to match path connections on any upper level that do not connect to current level.

export const isUpperLevelRoutingFilter: ExpressionFilterSpecification = [
	'any',
	['all', ['>', ceilFromLevel, currentLevel], ['>', ceilToLevel, currentLevel]],
	['all', ['>', floorFromLevel, currentLevel], ['>', floorToLevel, currentLevel]]
];

/// Filter to match paths that act as a connection from the current level to the upper level.

export const leadsToUpperLevelRoutingFilter: ExpressionFilterSpecification = [
	'all',
	['any', ['==', ceilFromLevel, currentLevel], ['==', floorFromLevel, currentLevel]],
	['any', ['>', ceilToLevel, currentLevel], ['>', floorToLevel, currentLevel]]
];

/// Filter to match paths that act as a connection from the upper level to the current level.

export const leadsDownToCurrentLevelRoutingFilter: ExpressionFilterSpecification = [
	'all',
	['any', ['>', ceilFromLevel, currentLevel], ['>', floorFromLevel, currentLevel]],
	['any', ['==', ceilToLevel, currentLevel], ['==', floorToLevel, currentLevel]]
];

/// Filter to match paths that act as a connection from the current level to the lower level.

export const leadsToLowerLevelRoutingFilter: ExpressionFilterSpecification = [
	'all',
	['any', ['==', ceilFromLevel, currentLevel], ['==', floorFromLevel, currentLevel]],
	['any', ['<', ceilToLevel, currentLevel], ['<', floorToLevel, currentLevel]]
];

/// Filter to match paths that act as a connection from the lower level to the current level.

export const leadsUpToCurrentLevelRoutingFilter: ExpressionFilterSpecification = [
	'all',
	['any', ['<', ceilFromLevel, currentLevel], ['<', floorFromLevel, currentLevel]],
	['any', ['==', ceilToLevel, currentLevel], ['==', floorToLevel, currentLevel]]
];

// indoor tile layer \\

export const ceilLevel: ExpressionFilterSpecification = [
	'coalesce',
	['ceil', ['to-number', ['get', 'level']]],
	0
];
export const floorLevel: ExpressionFilterSpecification = [
	'coalesce',
	['floor', ['to-number', ['get', 'level']]],
	0
];

/// Filter to only show element if level matches current level.
///
/// This will show features with level 0.5; 0.3; 0.7 on level 0 and on level 1

export const isCurrentLevelFilter: ExpressionFilterSpecification = [
	'any',
	['==', ceilLevel, currentLevel],
	['==', floorLevel, currentLevel]
];

/// Filter to match **any** level below the current level.

export const isLowerLevelFilter: ExpressionFilterSpecification = [
	// important that ceil and floor need to be lower
	'all',
	['<', ceilLevel, currentLevel],
	['<', floorLevel, currentLevel]
];
