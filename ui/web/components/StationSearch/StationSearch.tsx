import React from "react";

import { Translations } from "../App/Localization";
import { SearchInputField } from "../Overlay/SearchInputField";
import { Station } from "../Types/Connection";
import { Address } from "../Types/SuggestionTypes";

export const StationSearch: React.FC<{'translation': Translations, 'station': (Station | Address), 'setStationSearch': React.Dispatch<React.SetStateAction<(Station | Address)>>}> = (props) => {

    return(
        <div id="station-search" className="">
            <SearchInputField   translation={props.translation}
                                label={props.translation.search.destination}
                                station={props.station}
                                setSearchDisplay={props.setStationSearch}
                                localStorageStation='stationEvent'/>
        </div>
    );
};