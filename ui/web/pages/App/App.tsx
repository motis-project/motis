import React from 'react';

import { useRouter} from 'next/router';

import { Overlay } from '../Modules/Overlay';
import { Translations, deTranslations, enTranslations, plTranslations } from '../Modules/Localization';
import { StationSearch } from '../Modules/StationSearch';
import { MapContainer } from '../Modules/MapContainer';


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
declare global{
    interface Window {
        portEvents : any;
    }
}  


export const App: React.FC = () => {
    
    return (
        <div className='app'>
            {/* visible && <MapView />*/}
            <MapContainer translation={getQuery()}/>
            <Overlay translation={getQuery()}/>
            {//<StationSearchView />
            }
            <StationSearch />
        </div>
    );
};