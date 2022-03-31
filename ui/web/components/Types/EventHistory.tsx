import moment from 'moment';

import { Connection, Station, TripId } from './Connection';
import { Address } from './SuggestionTypes';

export interface SubOverlayEvent {
    id: string,
    station?: Station | Address,
    train?: Connection | TripId,
    stationTime?: moment.Moment,
}