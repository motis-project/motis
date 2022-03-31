import React from 'react';

import { useRouter } from 'next/router';
import moment from 'moment';
import { isMobile } from "react-device-detect";

import { Overlay } from '../Overlay/Overlay';
import { Translations, deTranslations, enTranslations, plTranslations } from './Localization';
import { StationSearch } from '../StationSearch/StationSearch';
import { MapContainer } from '../Map/MapContainer';
import { Interval } from '../Types/RoutingTypes';
import { elmAPIResponse } from '../Types/IntermodalRoutingTypes';
import { ScheduleInfoResponse } from '../Types/ScheduleInfo';
import { Connection, Station, TripId } from '../Types/Connection';
import { Address } from '../Types/SuggestionTypes';
import { SubOverlayEvent } from '../Types/EventHistory';

declare global {
    interface Window {
        portEvents: any; //used for communication with map libaries
    }
}


const getQuery = (): Translations => {
    let router = useRouter();
    let { lang } = router.query;
    if (lang === 'de') {
        return deTranslations;
    } else if (lang === 'pl') {
        return plTranslations;
    }
    return enTranslations;
}


export const App: React.FC = () => {

    // Hold the available Interval for Scheduling Information
    const [scheduleInfo, setScheduleInfo] = React.useState<Interval>(null);

    // Overlay and StationSearch communicate via this State
    const [stationSearch, setStationSearch] = React.useState<Station | Address>({ id: '', name: '' });

    // Current Date
    const [searchDate, setSearchDate] = React.useState<moment.Moment>(null);

    // Store identifier for currently displayed SubOverlay Content. Will be used as a stack to realize a history of content.
    const [subOverlayContent, setSubOverlayContent] = React.useState<SubOverlayEvent[]>([]);

    // Current hovered map Data
    const [mapData, setMapData] = React.useState<any>();

    //if the mouse is over the map the data for the hovered elements is set
    React.useEffect(() => {
        window.portEvents.sub('mapSetTooltip', function(data){
            setMapData(data);
        });
    });

    React.useEffect(() => {
        let requestURL = 'https://europe.motis-project.de/?elm=requestScheduleInfo';

        fetch(requestURL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content: {}, content_type: 'MotisNoMessage', destination: { target: '/lookup/schedule_info', type: 'Module' } })
        })
            .then(res => res.json())
            .then((res: elmAPIResponse) => {
                let intv = { begin: (res.content as ScheduleInfoResponse).begin, end: (res.content as ScheduleInfoResponse).end }
                let intvBegin = moment.unix(intv.begin);
                intvBegin.hour(moment().hour());
                intvBegin.minute(moment().minute());
                intvBegin.second(moment().second());
                setScheduleInfo(intv);
                setSearchDate(intvBegin);
            })
    }, []);

    React.useEffect(() => {
        if((stationSearch as Station).id !== ''){
            setSubOverlayContent([...subOverlayContent, {id: 'stationEvent', station: stationSearch, stationTime: moment()}]);
        }
    }, [stationSearch]);

    return (
        <div className='app'>
            {isMobile ?
                <Overlay translation={getQuery()} scheduleInfo={scheduleInfo} searchDate={searchDate} mapData={mapData} subOverlayContent={subOverlayContent} setSubOverlayContent={setSubOverlayContent} />
                :
                <>
                    <MapContainer translation={getQuery()} scheduleInfo={scheduleInfo} searchDate={searchDate} mapData={mapData} subOverlayContent={subOverlayContent} setSubOverlayContent={setSubOverlayContent}/>
                    <Overlay translation={getQuery()} scheduleInfo={scheduleInfo} searchDate={searchDate} mapData={mapData} subOverlayContent={subOverlayContent} setSubOverlayContent={setSubOverlayContent}/>
                    <StationSearch translation={getQuery()} station={stationSearch} setStationSearch={setStationSearch} />
                </>
            }
        </div>
    );
};