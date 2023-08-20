package de.motis_project.app2.journey;

import com.google.flatbuffers.FlatBufferBuilder;

import motis.Attribute;
import motis.Connection;
import motis.EventInfo;
import motis.Move;
import motis.MoveWrapper;
import motis.Position;
import motis.Range;
import motis.Station;
import motis.Stop;
import motis.Transport;
import motis.Trip;
import motis.TripId;
import motis.Walk;

public class CopyConnection {
    public static FlatBufferBuilder copyConnection(Connection c) {
        FlatBufferBuilder fbb = new FlatBufferBuilder();

        int stops = copyStops(fbb, c);
        int transports = copyMoves(fbb, c);
        int attributes = copyAttributes(fbb, c);
        int trips = copyTrips(fbb, c);

        Connection.startConnection(fbb);
        Connection.addStops(fbb, stops);
        Connection.addTransports(fbb, transports);
        Connection.addAttributes(fbb, attributes);
        Connection.addTrips(fbb, trips);
        fbb.finish(Connection.endConnection(fbb));

        return fbb;
    }

    /**
     * TRIPS
     **/
    private static int copyTrips(FlatBufferBuilder fbb, Connection c) {
        int[] attrArray = new int[c.tripsLength()];
        for (int i = 0; i < c.tripsLength(); i++) {
            Trip trip = c.trips(i);
            TripId id = trip.id();

            int x = fbb.createString(id.id());
            int stationId = fbb.createString(id.stationId());
            int targetStationId = fbb.createString(id.targetStationId());
            int lineId = fbb.createString(id.lineId());
            int tripId = TripId.createTripId(
                    fbb, x,
                    stationId, id.trainNr(), id.time(),
                    targetStationId, id.targetTime(), lineId);

            Trip.startTrip(fbb);
            Trip.addRange(fbb, Range.createRange(fbb, trip.range().from(), trip.range().to()));
            Trip.addId(fbb, tripId);
            attrArray[i] = Attribute.endAttribute(fbb);
        }
        return Connection.createTripsVector(fbb, attrArray);
    }

    /**
     * ATTRIBUTES
     **/
    private static int copyAttributes(FlatBufferBuilder fbb, Connection c) {
        int[] attrArray = new int[c.attributesLength()];
        for (int i = 0; i < c.attributesLength(); i++) {
            Attribute attr = c.attributes(i);

            int code = fbb.createString(attr.code());
            int text = fbb.createString(attr.text());

            Attribute.startAttribute(fbb);
            Attribute.addRange(fbb, Range.createRange(fbb, attr.range().from(), attr.range().to()));
            Attribute.addCode(fbb, code);
            Attribute.addText(fbb, text);
            attrArray[i] = Attribute.endAttribute(fbb);
        }
        return Connection.createAttributesVector(fbb, attrArray);
    }

    /**
     * STOPS
     **/
    private static int copyStops(FlatBufferBuilder fbb, Connection c) {
        int[] stopsArray = new int[c.stopsLength()];
        for (int i = 0; i < c.stopsLength(); i++) {
            Stop s = c.stops(i);
            int station = copyStation(fbb, s.station());
            int arr = copyEventInfo(fbb, s.arrival());
            int dep = copyEventInfo(fbb, s.departure());
            stopsArray[i] = Stop.createStop(fbb, station, arr, dep, s.exit(), s.enter());
        }
        return Connection.createStopsVector(fbb, stopsArray);
    }

    private static int copyStation(FlatBufferBuilder fbb, Station s) {
        int id = fbb.createString(s.id());
        int name = fbb.createString(s.name());
        int pos = Position.createPosition(fbb, s.pos().lat(), s.pos().lng());

        Station.startStation(fbb);
        Station.addPos(fbb, pos);
        Station.addId(fbb, id);
        Station.addName(fbb, name);
        return Station.endStation(fbb);
    }

    private static int copyEventInfo(FlatBufferBuilder fbb, EventInfo ev) {
        int track = fbb.createString(ev.track());

        EventInfo.startEventInfo(fbb);
        EventInfo.addReason(fbb, ev.reason());
        EventInfo.addScheduleTime(fbb, ev.scheduleTime());
        EventInfo.addTime(fbb, ev.time());
        EventInfo.addTrack(fbb, track);
        return EventInfo.endEventInfo(fbb);
    }

    /**
     * MOVES
     **/
    private static int copyMoves(FlatBufferBuilder fbb, Connection c) {
        int[] movesArray = new int[c.transportsLength()];
        for (int i = 0; i < c.transportsLength(); i++) {
            movesArray[i] = copyMove(fbb, c.transports(i));
        }
        return Connection.createTransportsVector(fbb, movesArray);
    }

    private static int copyMove(FlatBufferBuilder fbb, MoveWrapper move) {
        int moveCopy = 0;
        switch (move.moveType()) {
            case Move.Transport: {
                Transport t = new Transport();
                move.move(t);

                int name = fbb.createString(t.name());
                int direction = fbb.createString(t.direction());
                int lineId = fbb.createString(t.lineId());
                int provider = fbb.createString(t.provider());

                Transport.startTransport(fbb);
                Transport.addRange(fbb, Range.createRange(fbb, t.range().from(), t.range().to()));
                Transport.addName(fbb, name);
                Transport.addClasz(fbb, t.clasz());
                Transport.addDirection(fbb, direction);
                Transport.addLineId(fbb, lineId);
                Transport.addProvider(fbb, provider);
                moveCopy = Transport.endTransport(fbb);

                break;
            }
            case Move.Walk: {
                Walk w = new Walk();
                move.move(w);

                int mumoType = fbb.createString(w.mumoType());

                Walk.startWalk(fbb);
                Walk.addRange(fbb, Range.createRange(fbb, w.range().from(), w.range().to()));
                Walk.addMumoId(fbb, w.mumoId());
                Walk.addMumoType(fbb, mumoType);
                Walk.addPrice(fbb, w.price());
                moveCopy = Walk.endWalk(fbb);
                break;
            }
        }
        return MoveWrapper.createMoveWrapper(fbb, move.moveType(), moveCopy);
    }
}
