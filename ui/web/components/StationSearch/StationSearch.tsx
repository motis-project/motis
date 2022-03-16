import React from "react";

import { Translations } from "../App/Localization";
import { SearchInputField } from "../Overlay/SearchInputField";
import { Station } from "../Types/Connection";
import { Address } from "../Types/SuggestionTypes";

export const StationSearch: React.FC<{'translation': Translations}> = (props) => {

    const [station, setStation] = React.useState<Station | Address>({id: '', name: ''});

    return(
        <div id="station-search" className="">
            <SearchInputField   translation={props.translation}
                                            label={props.translation.search.destination}
                                            station={station}
                                            setSearchDisplay={setStation}
                                            localStorageStation='motis.routing.to_location'/>
        </div>
    );
};