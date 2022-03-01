window.portEvents = {
    events : {},
    sub : function (portName, callback){
        this.events[portName] = this.events[portName] || [];
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

const ownPorts = {
    'setRoutingResponses': {
        send: function(){}
    },
    'showStationDetails': {
        send: function(){}
    },
    'showTripDetails': {
        send: function(){}
    },
    'setTimeOffset': {
        subscribe: function(){}
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
        subscribe: function(){}
    },
    'mapInit': {
        subscribe: function(callback){
            window.portEvents.sub('mapInit', callback);
            window.portEvents.pub('mapInit', 'map'); //sollte wo anders hin
            window.portEvents.pub('mapInitFinished', true);
        }
    },
    'mapUpdate': {
        send: function(){}
    },
    'mapSetTooltip': {
        send: function(){}
    },
    'mapFlyTo': {
        subscribe: function(){}
    },
    'mapFitBounds': {
        subscribe: function(){}
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
        subscribe: function(){}
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
        subscribe: function(){}
    }
}