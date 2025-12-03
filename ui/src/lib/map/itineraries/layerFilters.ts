// routing layer \\

export const _currentLevel = ['coalesce', ['get', 'level'], 0];
export const _ceilFromLevel = ['coalesce', ['ceil', ['to-number', ['get', 'fromLevel']]], 0];
export const _ceilToLevel = ['coalesce', ['ceil', ['to-number', ['get', 'toLevel']]], 0];
export const _floorFromLevel = ['coalesce', ['floor', ['to-number', ['get', 'fromLevel']]], 0];
export const _floorToLevel = ['coalesce', ['floor', ['to-number', ['get', 'toLevel']]], 0];

/// Filter to match all connections that lie on, cross or connect to the current level.

export const _connectsToCurrentLevelRoutingFilter = [
	'all',
	['<=', ['min', _floorToLevel, _floorFromLevel], _currentLevel],
	['>=', ['max', _ceilToLevel, _ceilFromLevel], _currentLevel]
];

/// Filter to match path connections on the current level.

export const _isCurrentLevelRoutingFilter = [
	'any',
	['all', ['==', _ceilFromLevel, _currentLevel], ['==', _ceilToLevel, _currentLevel]],
	['all', ['==', _floorFromLevel, _currentLevel], ['==', _floorToLevel, _currentLevel]]
];

/// Filter to match path connections on any lower level that do not connect to current level.

export const _isLowerLevelRoutingFilter = [
	'any',
	['all', ['<', _ceilFromLevel, _currentLevel], ['<', _ceilToLevel, _currentLevel]],
	['all', ['<', _floorFromLevel, _currentLevel], ['<', _floorToLevel, _currentLevel]]
];

/// Filter to match path connections on any upper level that do not connect to current level.

export const _isUpperLevelRoutingFilter = [
	'any',
	['all', ['>', _ceilFromLevel, _currentLevel], ['>', _ceilToLevel, _currentLevel]],
	['all', ['>', _floorFromLevel, _currentLevel], ['>', _floorToLevel, _currentLevel]]
];

/// Filter to match paths that act as a connection from the current level to the upper level.

export const _leadsToUpperLevelRoutingFilter = [
	'all',
	['any', ['==', _ceilFromLevel, _currentLevel], ['==', _floorFromLevel, _currentLevel]],
	['any', ['>', _ceilToLevel, _currentLevel], ['>', _floorToLevel, _currentLevel]]
];

/// Filter to match paths that act as a connection from the upper level to the current level.

export const _leadsDownToCurrentLevelRoutingFilter = [
	'all',
	['any', ['>', _ceilFromLevel, _currentLevel], ['>', _floorFromLevel, _currentLevel]],
	['any', ['==', _ceilToLevel, _currentLevel], ['==', _floorToLevel, _currentLevel]]
];

/// Filter to match paths that act as a connection from the current level to the lower level.

export const _leadsToLowerLevelRoutingFilter = [
	'all',
	['any', ['==', _ceilFromLevel, _currentLevel], ['==', _floorFromLevel, _currentLevel]],
	['any', ['<', _ceilToLevel, _currentLevel], ['<', _floorToLevel, _currentLevel]]
];

/// Filter to match paths that act as a connection from the lower level to the current level.

export const _leadsUpToCurrentLevelRoutingFilter = [
	'all',
	['any', ['<', _ceilFromLevel, _currentLevel], ['<', _floorFromLevel, _currentLevel]],
	['any', ['==', _ceilToLevel, _currentLevel], ['==', _floorToLevel, _currentLevel]]
];

// indoor tile layer \\

export const _ceilLevel = ['coalesce', ['ceil', ['to-number', ['get', 'level']]], 0];
export const _floorLevel = ['coalesce', ['floor', ['to-number', ['get', 'level']]], 0];

/// Filter to only show element if level matches current level.
///
/// This will show features with level 0.5; 0.3; 0.7 on level 0 and on level 1

export const _isCurrentLevelFilter = [
	'any',
	['==', _ceilLevel, _currentLevel],
	['==', _floorLevel, _currentLevel]
];

/// Filter to match **any** level below the current level.

export const _isLowerLevelFilter = [
	// important that ceil and floor need to be lower
	'all',
	['<', _ceilLevel, _currentLevel],
	['<', _floorLevel, _currentLevel]
];
