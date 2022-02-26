import React, { useState, useEffect } from 'react';

import { getFromLocalStorage, setLocalStorage } from './LocalStorage';

export let markerSearch = null;

export const RailvizContextMenu: React.FC = () => {

    const [x, setX] = useState(0);
    const [y, setY] = useState(0);
    const [lat, setLat] = useState(0);
    const [lng, setLng] = useState(0);
    const [contextMenuHidden, setContextMenuHidden] = useState<Boolean>(true);

    useEffect(() => {
        window.portEvents.sub('mapShowContextMenu', function(data){
            setX(data.mouseX);
            setY(data.mouseY);
            setLat(data.lat);
            setLng(data.lng);
            setContextMenuHidden(false);
        });
        window.portEvents.sub('mapCloseContextMenu', function(data){
            setContextMenuHidden(true);
        })
    })
    return (
        <div className={contextMenuHidden ? 'railviz-contextmenu hidden': 'railviz-contextmenu'} style={{ top: y+'px', left: x+'px' }}>
            <div className='item' onClick={() => {
                setContextMenuHidden(true);
                setLocalStorage("motis.routing.from_location", {'name': lat+';'+lng, 'pos':{'lat': lat, 'lng': lng}, 'type_': '', 'regions': {}});
                markerSearch = [true, {'name': lat+';'+lng, 'pos':{'lat': lat, 'lng': lng}, 'type_': '', 'regions': {}}];
                window.portEvents.pub('mapSetMarkers', { 'startPosition':{'lat': lat,'lng': lng},
                                                         'startName': lat+';'+lng,
                                                         'destinationPosition': getFromLocalStorage("motis.routing.to_location").pos,
                                                         'destinationName': getFromLocalStorage("motis.routing.to_location").name});
                markerSearch = null;
            }}>Routen von hier</div>
            <div className='item' onClick={() => {
                setContextMenuHidden(true);
                setLocalStorage("motis.routing.to_location", {'name': lat+';'+lng, 'pos':{'lat': lat, 'lng': lng}, 'type_': '', 'regions': {}})
                markerSearch = [false, {'name': lat+';'+lng, 'pos':{'lat': lat, 'lng': lng}, 'type_': '', 'regions': {}}];
                window.portEvents.pub('mapSetMarkers', {'startPosition': getFromLocalStorage("motis.routing.from_location").pos,
                                                        'startName': getFromLocalStorage("motis.routing.from_location").name,
                                                        'destinationPosition':{'lat': lat, 'lng': lng},
                                                        'destinationName':lat+';'+lng});
                markerSearch = null;
            }}>Routen hierher</div>
        </div>);
};