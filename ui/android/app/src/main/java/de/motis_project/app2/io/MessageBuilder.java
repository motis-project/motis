package de.motis_project.app2.io;

import com.google.flatbuffers.FlatBufferBuilder;

import java.nio.ByteBuffer;
import java.util.Date;

import motis.Destination;
import motis.DestinationType;
import motis.Interval;
import motis.Message;
import motis.MotisError;
import motis.MotisNoMessage;
import motis.MsgContent;
import motis.Position;
import motis.guesser.StationGuesserRequest;
import motis.intermodal.FootPPR;
import motis.intermodal.InputPosition;
import motis.intermodal.IntermodalDestination;
import motis.intermodal.IntermodalPretripStart;
import motis.intermodal.IntermodalRoutingRequest;
import motis.intermodal.IntermodalStart;
import motis.intermodal.Mode;
import motis.intermodal.ModeWrapper;
import motis.ppr.FootRoutingRequest;
import motis.ppr.SearchDirection;
import motis.ppr.SearchOptions;
import motis.routing.InputStation;
import motis.routing.PretripStart;
import motis.routing.SearchDir;
import motis.routing.SearchType;

public class MessageBuilder {
    public static Message error(int ssid, int code, String category,
                                String reason) {
        FlatBufferBuilder b = new FlatBufferBuilder();
        int error = MotisError
                .createMotisError(b, code, b.createString(category),
                        b.createString(reason));
        b.finish(Message.createMessage(b, 0, MsgContent.MotisError, error,
                ssid));
        return Message.getRootAsMessage(b.dataBuffer());
    }

    public static byte[] guess(int ssid, String input) {
        FlatBufferBuilder b = new FlatBufferBuilder();
        int guesserRequestOffset = StationGuesserRequest
                .createStationGuesserRequest(b, 10, b.createString(input));
        int destination = Destination.createDestination(
                b, DestinationType.Module, b.createString("/guesser"));
        b.finish(Message.createMessage(
                b, destination, MsgContent.StationGuesserRequest,
                guesserRequestOffset, ssid));
        return b.sizedByteArray();
    }

    public static byte[] route(
            int ssid,
            String fromId, String toId,
            boolean isArrival,
            Date intervalBegin, Date intervalEnd,
            boolean extendIntervalEarlier,
            boolean extendIntervalLater,
            int minConnectionCount) {
        FlatBufferBuilder b = new FlatBufferBuilder();

        String startStationId = isArrival ? toId : fromId;
        String targetStationId = isArrival ? fromId : toId;

        int start = createPreTripStart(
                b, startStationId,
                intervalBegin, intervalEnd,
                extendIntervalEarlier, extendIntervalLater,
                minConnectionCount);
        int destination = InputStation.createInputStation(
                b, b.createString(targetStationId), b.createString(""));
        int router = b.createString("");
        int intermodalRoutingRequest = IntermodalRoutingRequest.createIntermodalRoutingRequest(b,
                IntermodalStart.PretripStart, start,
                IntermodalRoutingRequest.createStartModesVector(b, new int[]{}),
                IntermodalDestination.InputStation, destination,
                IntermodalRoutingRequest.createDestinationModesVector(b, new int[]{}),
                SearchType.Default,
                isArrival ? SearchDir.Backward : SearchDir.Forward, router
        );

        b.finish(Message.createMessage(
                b, Destination.createDestination(
                        b, DestinationType.Module, b.createString("/intermodal")),
                MsgContent.IntermodalRoutingRequest, intermodalRoutingRequest, ssid));

        return b.sizedByteArray();
    }

    public static byte[] scheduleInfo(int ssid) {
        FlatBufferBuilder b = new FlatBufferBuilder();

        MotisNoMessage.startMotisNoMessage(b);
        int noMsg = MotisNoMessage.endMotisNoMessage(b);

        b.finish(Message.createMessage(
                b, Destination.createDestination(
                        b, DestinationType.Module, b.createString("/lookup/schedule_info")),
                MsgContent.MotisNoMessage, noMsg, ssid));

        return b.sizedByteArray();
    }


    static private int createPreTripStart(
            FlatBufferBuilder b,
            String startStationId,
            Date intervalBegin, Date intervalEnd,
            boolean extendIntervalEarlier, boolean extendIntervalLater,
            int minConnectionCount) {
        int station = InputStation.createInputStation(
                b, b.createString(startStationId), b.createString(""));

        PretripStart.startPretripStart(b);
        PretripStart.addStation(b, station);
        PretripStart.addInterval(
                b, Interval.createInterval(
                        b, intervalBegin.getTime() / 1000,
                        intervalEnd.getTime() / 1000));
        PretripStart.addExtendIntervalEarlier(b, extendIntervalEarlier);
        PretripStart.addExtendIntervalLater(b, extendIntervalLater);
        PretripStart.addMinConnectionCount(b, minConnectionCount);
        return PretripStart.endPretripStart(b);
    }

    public static Message decode(byte[] buf) {
        return Message.getRootAsMessage(ByteBuffer.wrap(buf));
    }
}
