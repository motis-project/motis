window.portEvents = {
    events : {},
    sub : function (portName, callback){
        this.events[portName] = this.events[portName] || [];
        for(let i = 0; i < this.events[portName].length; i++){
            if(this.events[portName][i].toString() === callback.toString()){
                return;
            }
        }
        this.events[portName].push(callback);
    },
    unsub : function (portName, callback){
        if(this.events[portName]){
            for(let i = 0; i < this.events[portName].length; i++){
                if(this.events[portName][i] === callback){
                    this.events[portName].splice(i, 1);
                    break;
                }
            }
        }
    },

    pub : function (portName, data){
        if(this.events[portName]){
            this.events[portName].forEach(function (fn){
                fn(data);
            });
        }
    }
};

// ports that the map useses
const ownPorts = {
    'setRoutingResponses': {
        send: function(){}
    },
    'showStationDetails': {
        send: function(callback){
            window.portEvents.pub('showStationDetails', callback);
        }
    },
    'showTripDetails': {
        send: function(callback){
            window.portEvents.pub('showTripDetails', callback);
        }
    },
    'setTimeOffset': {
        subscribe: function(callback){
            window.portEvents.sub('setTimeOffset', callback);
        }
    },
    'setSimulationTime': {
        send: function(){}
    },
    'handleRailVizError': {
        send: function(){}
    },
    'clearRailVizError': {
        send: function(){}
    },
    'localStorageSet': {
        subscribe: function(){}
    },
    'setPPRSearchOptions': {
        subscribe: function(callback){
            window.portEvents.sub('setPPRSearchOptions', callback);
        }
    },
    'mapInit': {
        subscribe: function(callback){
            window.portEvents.sub('mapInit', callback);
            window.portEvents.pub('mapInit', 'map'); //sollte wo anders hin
            window.portEvents.pub('mapInitFinished', true);
        }
    },
    'mapUpdate': {
        send: function(callback){
            window.portEvents.pub('mapUpdate', callback);
        }
    },
    'mapSetTooltip': {
        send: function(callback){
            window.portEvents.pub('mapSetTooltip', callback);
        }
    },
    'mapFlyTo': {
        subscribe: function(callback){
            window.portEvents.sub('mapFlyTo', callback);
        }
    },
    'mapFitBounds': {
        subscribe: function(callback){
            window.portEvents.sub('mapFitBounds', callback);
        }
    },
    'mapUseTrainClassColors': {
        subscribe: function(callback){
            window.portEvents.sub('mapUseTrainClassColors', callback);
        }
    },
    'mapShowTrains': {
        subscribe: function(callback){
            window.portEvents.sub('mapShowTrains', callback);
        }
    },
    'mapSetDetailFilter': {
        subscribe: function(callback){
            window.portEvents.sub('mapSetDetailFilter', callback);
        }
    },
    'mapUpdateWalks': {
        subscribe: function(){}
    },
    'mapShowContextMenu': {
        send: function(callback){
            window.portEvents.pub('mapShowContextMenu', callback);
        }
    },
    'mapCloseContextMenu': {
        send: function(callback){
            window.portEvents.pub('mapCloseContextMenu', callback);
        }
    },
    'mapSetMarkers': {
        subscribe: function(callback){
            window.portEvents.sub('mapSetMarkers', callback);
        }
    },
    'mapSetLocale': {
        subscribe: function(callback){
            window.portEvents.sub('mapSetLocale', callback);
        }
    },
    'mapSetConnections': {
        subscribe: function(callback){
            window.portEvents.sub('mapSetConnections', callback);
        }
    },
    'mapHighlightConnections': {
        subscribe: function(callback){
            window.portEvents.sub('mapHighlightConnections', callback);
        }
    }
}