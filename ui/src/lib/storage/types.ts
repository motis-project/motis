import type {
	ElevationCosts,
	Mode,
	PedestrianProfile,
	PlanData
} from '@motis-project/motis-client';
import type { DisplayLevel } from '$lib/map/IsochronesShared';
import type { PrePostDirectMode } from '$lib/Modes';

/**
 * Interface for all user options that should be persisted in localStorage
 */
export interface SavedOptions {
	// Search options
	arriveBy: boolean;
	timetableView: boolean;
	searchWindow: number;
	numItineraries: number;
	algorithm: PlanData['query']['algorithm'];

	// Advanced options
	useRoutedTransfers: boolean;
	pedestrianProfile: PedestrianProfile;
	requireBikeTransport: boolean;
	requireCarTransport: boolean;
	transitModes: Mode[];
	preTransitModes: PrePostDirectMode[];
	postTransitModes: PrePostDirectMode[];
	directModes: PrePostDirectMode[] | undefined;
	elevationCosts: ElevationCosts;
	maxTransfers: number;
	maxPreTransitTime: number;
	maxPostTransitTime: number;
	maxDirectTime: number | undefined;
	ignorePreTransitRentalReturnConstraints: boolean;
	ignorePostTransitRentalReturnConstraints: boolean;
	ignoreDirectRentalReturnConstraints: boolean | undefined;
	preTransitProviderGroups: string[];
	postTransitProviderGroups: string[];
	directProviderGroups: string[];

	// Isochrones options
	isochronesDisplayLevel: DisplayLevel;
	isochronesColor: string;
	isochronesOpacity: number;
	isochronesCircleResolution: number | undefined;

	// Display options
	theme: 'light' | 'dark';
	colorMode: 'rt' | 'route' | 'mode' | 'none';
	showMap: boolean;

	// Map options
	center: [number, number];
	zoom: number;
	level: number;
}
