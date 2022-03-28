import React from 'react';

import moment from 'moment';

import { Translations } from '../App/Localization';
import { Station, TransportInfo, TripId } from '../Types/Connection';
import { TripView } from './TripView';
import { Interval } from '../Types/RoutingTypes';
import { StationEvent } from '../StationSearch/StationEvent';
import { Address } from '../Types/SuggestionTypes';
import { TripSearchHeader } from './TripSearchHeader';
import { TripSearch } from './TripSearch';
import { Trip } from '../Types/RailvizStationEvent';

export const SubOverlay: React.FC<{ 'translation': Translations, 'subOverlayHidden': Boolean, 'setSubOverlayHidden': React.Dispatch<React.SetStateAction<Boolean>>, 'trainSelected': TripId, 'setTrainSelected': React.Dispatch<React.SetStateAction<TripId>>, 'detailViewHidden': Boolean, 'scheduleInfo': Interval, 'stationEventTrigger': boolean, 'setStationEventTrigger': React.Dispatch<React.SetStateAction<boolean>>, 'station': (Station | Address), 'searchDate': moment.Moment, 'setStationSearch': React.Dispatch<React.SetStateAction<Station | Address>>, 'mapFilter': any}> = (props) => {

    const [trips, setTrips] = React.useState<{first_station: Station, trip_info: Trip}[]>(null);

    const [subOverlayDate, setSubOverlayDate] = React.useState<moment.Moment>(moment());

    const [trainNumber, setTrainNumber] = React.useState<string>('');

    React.useEffect(() => {
        if(props.searchDate) {
            setSubOverlayDate(props.searchDate);
        }
    }, [props.searchDate]);

    return (
        <div className={props.subOverlayHidden ? 'sub-overlay hidden' : 'sub-overlay'}>
            <div id='sub-overlay-content'>
                {(props.stationEventTrigger) ?
                    <StationEvent translation={props.translation} station={props.station} stationEventTrigger={props.stationEventTrigger} setSubOverlayHidden={props.setSubOverlayHidden} setStationEventTrigger={props.setStationEventTrigger} searchDate={subOverlayDate}/> 
                    :
                    (props.trainSelected === undefined) ?
                        <div className='trip-search'>
                            <TripSearchHeader   translation={props.translation}
                                                searchDate={props.searchDate}
                                                scheduleInfo={props.scheduleInfo}
                                                trainNumber={trainNumber}
                                                setTrips={setTrips}
                                                setTrainNumber={setTrainNumber}/>
                            <div className='trips'>
                            {trips ? 
                                trips.map((trip: {first_station: Station, trip_info: Trip}) => 
                                    <TripSearch translation={props.translation} 
                                                trip={trip} 
                                                setTrainSelected={props.setTrainSelected}
                                                setStationSearch={props.setStationSearch}
                                                setSubOverlayDate={setSubOverlayDate}/>)
                                :
                                <></>
                            }
                            </div>
                        </div>
                        :
                        <TripView subOverlayHidden={props.subOverlayHidden} setSubOverlayHidden={props.setSubOverlayHidden} trainSelected={props.trainSelected} setTrainSelected={props.setTrainSelected} detailViewHidden={props.detailViewHidden} translation={props.translation} displayDate={props.searchDate} mapFilter={props.mapFilter}/>
                }
            </div>
            <div className='sub-overlay-close' onClick={() => {props.setTrainSelected(undefined); window.portEvents.pub('mapSetDetailFilter', props.mapFilter); props.setSubOverlayHidden(true); props.setStationEventTrigger(false)}}><i className='icon'>close</i></div>
        </div>
    )
}