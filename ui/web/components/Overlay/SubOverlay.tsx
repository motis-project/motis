import React from 'react';

import moment from 'moment';

import { Translations } from '../App/Localization';
import { Station, TripId } from '../Types/Connection';
import { TripView } from './TripView';
import { Interval } from '../Types/RoutingTypes';
import { StationEvent } from '../StationSearch/StationEvent';
import { TripSearchHeader } from './TripSearchHeader';
import { TripSearch } from './TripSearch';
import { Trip } from '../Types/RailvizStationEvent';
import { SubOverlayEvent } from '../Types/EventHistory';


interface SubOverlay {
    'translation': Translations,
    'trainSelected': TripId,
    'scheduleInfo': Interval,
    'searchDate': moment.Moment,
    'subOverlayContent': SubOverlayEvent[], 
    'setTrainSelected': React.Dispatch<React.SetStateAction<TripId>>,
    'setSubOverlayContent': React.Dispatch<React.SetStateAction<SubOverlayEvent[]>>,
    'setSubOverlayToggle': React.Dispatch<React.SetStateAction<boolean>>,
    'mapFilter': any
}


export const SubOverlay: React.FC<SubOverlay> = (props) => {

    const [trips, setTrips] = React.useState<{first_station: Station, trip_info: Trip}[]>(null);

    const [trainNumber, setTrainNumber] = React.useState<string>('');

    return (
        <div className={props.subOverlayContent.length === 0 ? 'sub-overlay hidden' : 'sub-overlay'}>
            <div id='sub-overlay-content'>
                { props.subOverlayContent.length !== 0 ? 
                    {
                        'tripSearch': 
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
                                                    subOverlayContent={props.subOverlayContent} 
                                                    setSubOverlayContent={props.setSubOverlayContent}
                                                    setTrainSelected={props.setTrainSelected}/>)
                                    :
                                    <></>
                                }
                                </div>
                            </div>,
                        'stationEvent':
                            <StationEvent translation={props.translation} station={props.subOverlayContent.at(-1).station} searchDate={props.subOverlayContent.at(-1).stationTime} setTrainSelected={props.setTrainSelected} subOverlayContent={props.subOverlayContent} setSubOverlayContent={props.setSubOverlayContent}/> ,
                        'tripView':
                            <TripView   trainSelected={props.subOverlayContent.at(-1).train}
                                        setTrainSelected={props.setTrainSelected}
                                        setTripViewHidden={null}
                                        translation={props.translation}
                                        subOverlayContent={props.subOverlayContent} 
                                        setSubOverlayContent={props.setSubOverlayContent}
                                        mapFilter={props.mapFilter}/>
                    }[props.subOverlayContent.at(-1).id]
                    :
                    <></>
                }
            </div>
            <div className='sub-overlay-close' onClick={() => {
                                                    props.setSubOverlayContent([]);
                                                    props.setSubOverlayToggle(false);
                                                    window.portEvents.pub('mapSetDetailFilter', props.mapFilter);
                                                    }}>
                <i className='icon'>close</i>
            </div>
        </div>
    )
}