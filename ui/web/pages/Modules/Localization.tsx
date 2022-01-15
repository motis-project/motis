interface SearchProfileNames {
      default : String
    , accessibility1 : String
    , wheelchair : String
    , elevation : String
    , custom : String
}

export interface Translations { 
    search :
        { search : String
        , start : String
        , destination : String
        , date : String
        , time : String
        , startTransports : String
        , destinationTransports : String
        , departure : String
        , arrival : String
        , trainNr : String
        , maxDuration : String
        , searchProfile :
            { profile : String
            , walkingSpeed : String
            , stairsUp : String
            , stairsDown : String
            , stairsWithHandrailUp : String
            , stairsWithHandrailDown : String
            , timeCost : String
            , accessibilityCost : String
            , streetCrossings : String
            , signals : String
            , marked : String
            , island : String
            , unmarked : String
            , primary : String
            , secondary : String
            , tertiary : String
            , residential : String
            , elevationUp : String
            , elevationDown : String
            , roundAccessibility : String
            , elevators : String
            , escalators : String
            , movingWalkways : String
            }
        , useParking : String
        },
    connections :
        { timeHeader : String
        , durationHeader : String
        , transportsHeader : String
        , scheduleRange(begin: Date, end: Date) : String
        , loading : String
        , noResults : String
        , extendBefore : String
        , extendAfter : String
        , interchanges(count: number) : String
        , walkDuration(duration: String) : String
        , interchangeDuration(duration: String) : String
        , arrivalTrack(track: String) : String
        , track : String
        , tripIntermediateStops(count: number) : String
        , tripWalk(duration: String) : String
        , tripBike(duration: String) : String
        , tripCar(duration: String) : String
        , provider : String
        , walk : String
        , bike : String
        , car : String
        , trainNr : String
        , lineId : String
        , parking : String
        },
    station :
        { direction : String
        , noDepartures : String
        , noArrivals : String
        , loading : String
        , trackAbbr : String
        },
    railViz :
        { noTrains : String
        , delayColors : String
        , classColors : String
        , simActive : String
        },
    mapContextMenu :
        { routeFromHere : String
        , routeToHere : String
        },
    errors :
        { journeyDateNotInSchedule : String
        , internalError(msg: String) : String
        , timeout : String
        , network : String
        , http(code: number) : String
        , decode(msg: String) : String
        , moduleNotFound : String
        , osrmProfileNotAvailable : String
        , osrmNoRoutingResponse : String
        , pprProfileNotAvailable : String
        },
    trips :
        { noResults : String
        },
    misc :
        { permalink : String },
    simTime :
        { simMode : String
        },
    searchProfiles :
        SearchProfileNames
}


const enDateConfig = (date: Date) => {
    let res = new Date(date);
    let dd = String(res.getDate()).padStart(2, '0');
    let mm = String(res.getMonth() + 1).padStart(2, '0'); //January is 0!
    let yy = String(res.getFullYear()).substring(-2);
    return dd + '/' + mm + '/' + yy
}


const deDateConfig = (date: Date) => {
    let res = new Date(date);
    let dd = String(res.getDate()).padStart(2, '0');
    let mm = String(res.getMonth() + 1).padStart(2, '0'); //January is 0!
    let yy = String(res.getFullYear()).substring(-2);
    return dd + '.' + mm + '.' + yy
}


export const deTranslations: Translations = {
        search:
            { search : "Suchen"
            , start : "Start"
            , destination : "Ziel"
            , date : "Datum"
            , time : "Uhrzeit"
            , startTransports : "Verkehrsmittel am Start"
            , destinationTransports : "Verkehrsmittel am Ziel"
            , departure : "Abfahrt"
            , arrival : "Ankunft"
            , trainNr : "Zugnummer"
            , maxDuration : "Maximale Dauer (Minuten)"
            , searchProfile :
                { profile : "Profil"
                , walkingSpeed : "Laufgeschwindigkeit (m/s)"
                , stairsUp : "Treppen (aufwärts)"
                , stairsDown : "Treppen (abwärts)"
                , stairsWithHandrailUp : "Treppen mit Geländer (aufwärts)"
                , stairsWithHandrailDown : "Treppen mit Geländer (abwärts)"
                , timeCost : "Zeitaufwand"
                , accessibilityCost : "Beschwerlichkeit"
                , streetCrossings : "Straßenüberquerungen"
                , signals : "Ampeln"
                , marked : "Zebrastreifen"
                , island : "Verkehrsinseln"
                , unmarked : "Unmarkiert"
                , primary : "Primary"
                , secondary : "Secondary"
                , tertiary : "Tertiary"
                , residential : "Residential"
                , elevationUp : "Höhenunterschiede (aufwärts)"
                , elevationDown : "Höhenunterschiede (abwärts)"
                , roundAccessibility : "Rundung Beweglichkeit"
                , elevators : "Aufzüge"
                , escalators : "Rolltreppen"
                , movingWalkways : "Fahrsteige"
                }
            , useParking : "Parkplätze verwenden"
            }
        , connections :
            { timeHeader : "Zeit"
            , durationHeader : "Dauer"
            , transportsHeader : "Verkehrsmittel"
            , scheduleRange :
                (begin: Date, end: Date) => 'Auskunft von ' 
                                + deDateConfig(begin) 
                                + ' bis ' 
                                + deDateConfig(end) 
                                + ' möglich'
            , loading : "Verbindungen suchen..."
            , noResults : "Keine Verbindungen gefunden"
            , extendBefore : "Früher"
            , extendAfter : "Später"
            , interchanges :
                (count: number) => count == 0 ? 'Keine Umstiege' : count == 1 ? '1 Umstieg' : count + 'Umstiege'
            , walkDuration : (duration: string) => duration + 'Fußweg'
            , interchangeDuration : (duration: string) => duration + 'Umstieg'
            , arrivalTrack : (track: string) => 'Ankunft Gleis' + track
            , track : "Gleis"
            , tripIntermediateStops :
                (count: number) => count == 0 ? 'Fahrt ohne Zwischenhalt' : count == 1 ? 'Fahrt 1 Station' : 'Fahrt ' + count + ' Stationen'
            , tripWalk : (duration: string) => 'Fußweg (' + duration + ')'
            , tripBike : (duration: string) => 'Fahrrad (' + duration + ')'
            , tripCar : (duration: string) => 'Auto (' + duration + ')'
            , provider : "Betreiber"
            , walk : "Fußweg"
            , bike : "Fahrrad"
            , car : "Auto"
            , trainNr : "Zugnummer"
            , lineId : "Linie"
            , parking : "Parkplatz"
            }
        , station :
            { direction : "Richtung"
            , noDepartures : "Keine Abfahrten im gewählten Zeitraum"
            , noArrivals : "Keine Ankünfte im gewählten Zeitraum"
            , loading : "Laden..."
            , trackAbbr : "Gl."
            }
        , railViz :
            { noTrains : "Keine Züge"
            , delayColors : "Nach Verspätung"
            , classColors : "Nach Kategorie"
            , simActive : "Simulationsmodus aktiv"
            }
        , mapContextMenu :
            { routeFromHere : "Routen von hier"
            , routeToHere : "Routen hierher"
            }
        , errors :
            { journeyDateNotInSchedule : "Zeitraum nicht im Fahrplan"
            , internalError : (msg: string) => "Interner Fehler (" + msg + ")"
            , timeout : "Zeitüberschreitung"
            , network : "Netzwerkfehler"
            , http : (code: number) => "HTTP-Fehler " + code
            , decode : (msg: string) => "Ungültige Antwort (" + msg + ")"
            , moduleNotFound : "Modul nicht geladen"
            , osrmProfileNotAvailable : "OSRM: Profil nicht verfügbar"
            , osrmNoRoutingResponse : "OSRM: Keine Routing-Antwort"
            , pprProfileNotAvailable : "PPR: Profil nicht verfügbar"
            }
        , trips :
            { noResults : "Keine passenden Züge gefunden"
            }
        , misc :
            { permalink : "Permalink" }
        , simTime :
            { simMode : "Simulationsmodus"
            }
        , searchProfiles :
            { default : "Standard"
            , accessibility1 : "Auch nach leichten Wegen suchen"
            , wheelchair : "Rollstuhl"
            , elevation : "Weniger Steigung"
            , custom : "Benutzerdefiniert"
            }
        };


export const enTranslations: Translations = {
        search :
            { search : "Search"
            , start : "Start"
            , destination : "Destination"
            , date : "Date"
            , time : "Time"
            , startTransports : "Transports at the start"
            , destinationTransports : "Transports at the destination"
            , departure : "Departure"
            , arrival : "Arrival"
            , trainNr : "Train Number"
            , maxDuration : "Max. duration (minutes)"
            , searchProfile :
                { profile : "Profil"
                , walkingSpeed : "Walking speed (m/s)"
                , stairsUp : "Stairs (up)"
                , stairsDown : "Stairs (down)"
                , stairsWithHandrailUp : "Stairs with handrail (up)"
                , stairsWithHandrailDown : "Stairs with handrail (down)"
                , timeCost : "Time cost"
                , accessibilityCost : "Accessibility cost"
                , streetCrossings : "Street crossings"
                , signals : "Traffic signals"
                , marked : "Marked (zebra crossings)"
                , island : "Traffic islands"
                , unmarked : "Unmarked"
                , primary : "Primary"
                , secondary : "Secondary"
                , tertiary : "Tertiary"
                , residential : "Residential"
                , elevationUp : "Elevation difference (up)"
                , elevationDown : "Elevation difference (down)"
                , roundAccessibility : "Round accessibility"
                , elevators : "Elevators"
                , escalators : "Escalators"
                , movingWalkways : "Moving walkways"
                }
            , useParking : "Use parkings"
            }
        , connections :
            { timeHeader : "Time"
            , durationHeader : "Duration"
            , transportsHeader : "Transports"
            , scheduleRange :
                (begin: Date, end: Date) => 
                    'Possible dates: '
                    + enDateConfig(begin)
                    + ' - '
                    + enDateConfig(end)
            , loading : "Searching..."
            , noResults : "No connections found"
            , extendBefore : "Earlier"
            , extendAfter : "Later"
            , interchanges :
                (count: number) => count == 0 ? 'No interchanges' : count == 1 ? '1 Interchange' : count + ' interchanges'
            , walkDuration : (duration: string) => duration + " walk"
            , interchangeDuration : (duration: string) => duration + " interchange"
            , arrivalTrack : (track: string) => "Arrival track " + track
            , track : "Track"
            , tripIntermediateStops :
                (count: number) => count == 0 ? 'No intermediate stops' : count == 1 ? '1 intermediate stop' : count + ' intermediate stops'
            , tripWalk : (duration: string) => "Walk (" + duration + ")"
            , tripBike : (duration: string) => "Bike (" + duration + ")"
            , tripCar : (duration: string) => "Car (" + duration + ")"
            , provider : "Provider"
            , walk : "Walk"
            , bike : "Bike"
            , car : "Car"
            , trainNr : "Train number"
            , lineId : "Line"
            , parking : "Parking"
            }
        , station :
            { direction : "Direction"
            , noDepartures : "No departures"
            , noArrivals : "No arrivals"
            , loading : "Loading..."
            , trackAbbr : "Tr."
            }
        , railViz :
            { noTrains : "No trains"
            , delayColors : "By delay"
            , classColors : "By category"
            , simActive : "Simulation mode active"
            }
        , mapContextMenu :
            { routeFromHere : "Directions from here"
            , routeToHere : "Directions to here"
            }
        , errors :
            { journeyDateNotInSchedule : "Date not in schedule"
            , internalError : (msg: string) => "Internal error (" + msg + ")"
            , timeout : "Timeout"
            , network : "Network error"
            , http : (code: number) => "HTTP error " + code
            , decode : (msg: string) => "Invalid response (" + msg + ")"
            , moduleNotFound : "Module not found"
            , osrmProfileNotAvailable : "OSRM: Profile not available"
            , osrmNoRoutingResponse : "OSRM: No routing response"
            , pprProfileNotAvailable : "PPR: Profile not available"
            }
        , trips :
            { noResults : "No matching trains found"
            }
        , misc :
            { permalink : "Permalink" }
        , simTime :
            { simMode : "Simulation mode"
            }
        , searchProfiles :
            { default : "Default"
            , accessibility1 : "Include accessible routes"
            , wheelchair : "Wheelchair"
            , elevation : "Avoid elevation changes"
            , custom : "Custom"
            }
        };