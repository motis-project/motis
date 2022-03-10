import React from 'react';

import { useRouter} from 'next/router';

import { Overlay } from '../Overlay/Overlay';
import { Translations, deTranslations, enTranslations, plTranslations } from './Localization';
import { StationSearch } from '../StationSearch/StationSearch';
import { MapContainer } from '../Map/MapContainer';

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

    let isMobile = false;

    React.useEffect(() => {
        isMobile = window.matchMedia("only screen and (max-width: 500px)").matches;
    }, []);

    
    return (
        <div className='app'>
            {isMobile ?
                <Overlay translation={getQuery()}/>
                :
                <div>
                    {/* visible && <MapView />*/}
                    <MapContainer translation={getQuery()}/>
                    <Overlay translation={getQuery()}/>
                    {//<StationSearchView />}
                    }<StationSearch translation={getQuery()}/>
                </div>
            }
        </div>
    );
};