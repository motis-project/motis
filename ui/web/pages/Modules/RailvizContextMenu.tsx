import React, { useState } from 'react';

export const RailvizContextMenu: React.FC = () => {

    const [x, setX] = useState(0);
    const [y, setY] = useState(0);
    const [contextMenuHidden, setContextMenuHidden] = useState<Boolean>(true);

    React.useEffect(() => {
        window.portEvents.sub('mapShowContextMenu', function(data){
            setX(data.mouseX);
            setY(data.mouseY);
            setContextMenuHidden(false);
        });
        window.portEvents.sub('mapCloseContextMenu', function(data){
            setContextMenuHidden(true);
        })
    })
    return (
    <div className={contextMenuHidden ? 'railviz-contextmenu hidden': 'railviz-contextmenu'} style={{ top: y+'px', left: x+'px' }}>
        <div className='item' onClick={() => setContextMenuHidden(true)}>Routen von hier</div>
        <div className='item' onClick={() => setContextMenuHidden(true)}>Routen hierher</div>
    </div>);
};