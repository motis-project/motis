import React, { useState } from 'react';

export const RailvizContextMenu: React.FC = () => {

    const [x, setX] = useState(0);
    const [y, setY] = useState(0);
    const [lat, setLat] = useState(0);
    const [lng, setLng] = useState(0);
    const [contextMenuHidden, setContextMenuHidden] = useState<Boolean>(true);

    React.useEffect(() => {
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
        //wir brauchen hier start und Ziel positionen aus der Suche
        <div className={contextMenuHidden ? 'railviz-contextmenu hidden': 'railviz-contextmenu'} style={{ top: y+'px', left: x+'px' }}>
            <div className='item' onClick={() => {
                setContextMenuHidden(true);
                window.portEvents.pub('mapSetMarkers', {'startPosition':{'lat': lat,'lng': lng}, 'startName': lat+';'+lng, 'destinationPosition': null, 'destinationName': null});
            }}>Routen von hier</div>
            <div className='item' onClick={() => {
                setContextMenuHidden(true);
                window.portEvents.pub('mapSetMarkers', {'startPosition': null, 'startName': null, 'destinationPosition':{'lat': lat, 'lng': lng}, 'destinationName':lat+';'+lng});
            }}>Routen hierher</div>
        </div>);
};