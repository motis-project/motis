import React from 'react';

import { useRouter} from 'next/router';
import moment from 'moment';

import { Overlay } from '../Overlay/Overlay';
import { Translations, deTranslations, enTranslations, plTranslations } from './Localization';
import { StationSearch } from '../StationSearch/StationSearch';
import { MapContainer } from '../Map/MapContainer';
import { Interval } from '../Types/RoutingTypes';
import { elmAPIResponse } from '../Types/IntermodalRoutingTypes';
import { ScheduleInfoResponse } from '../Types/ScheduleInfo';

declare global{
    interface Window {
        portEvents : any;
    }
}  


const getQuery = (): Translations => {
    let router = useRouter();
    let { locale } = router.query;
    if (locale === 'de') {
        return deTranslations;
    } else if (locale === 'pl') {
        return plTranslations;
    }
    return enTranslations;
}


export const App: React.FC = () => {

    // Hold the available Interval for Scheduling Information
    const [scheduleInfo, setScheduleInfo] = React.useState<Interval>(null);
    
    let isMobile = false;

    React.useEffect(() => {
        isMobile = window.matchMedia("only screen and (max-width: 500px)").matches;
    }, []);

    React.useEffect(() => {
        let requestURL = 'https://europe.motis-project.de/?elm=requestScheduleInfo';

        fetch(requestURL, { method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({content: {}, content_type: 'MotisNoMessage', destination: { target: '/lookup/schedule_info', type: 'Module' }})})
        .then(res => res.json())
        .then((res: elmAPIResponse) => {
            console.log("Response came in");
            console.log(res);
            let intv = {begin: (res.content as ScheduleInfoResponse).begin, end: (res.content as ScheduleInfoResponse).end}
            let intvBegin = moment.unix(intv.begin);
            intvBegin.hour(moment().hour());
            intvBegin.minute(moment().minute());
            setScheduleInfo(intv);
        })
    }, []);
    
    return (
        <div className='app'>
            {isMobile ?
                <Overlay translation={getQuery()} scheduleInfo={scheduleInfo}/>
                :
                <>
                    {/* visible && <MapView />*/}
                    <MapContainer translation={getQuery()} scheduleInfo={scheduleInfo}/>
                    <Overlay translation={getQuery()} scheduleInfo={scheduleInfo}/>
                    {//<StationSearchView />}
                    }<StationSearch translation={getQuery()}/>
                </>
            }
        </div>
    );
};