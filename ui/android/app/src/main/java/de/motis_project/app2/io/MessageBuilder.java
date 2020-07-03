package de.motis_project.app2.io;

import com.google.flatbuffers.FlatBufferBuilder;

import java.nio.ByteBuffer;
import java.util.Date;

import de.motis_project.app2.intermodal.IntermodalQuery;
import de.motis_project.app2.ppr.NamedLocation;
import de.motis_project.app2.ppr.profiles.PprSearchOptions;
import de.motis_project.app2.ppr.query.PPRQuery;
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
        int intermodalRoutingRequest = IntermodalRoutingRequest.createIntermodalRoutingRequest(b,
                IntermodalStart.PretripStart, start,
                IntermodalRoutingRequest.createStartModesVector(b, new int[]{}),
                IntermodalDestination.InputStation, destination,
                IntermodalRoutingRequest.createDestinationModesVector(b, new int[]{}),
                SearchType.Default,
                isArrival ? SearchDir.Backward : SearchDir.Forward
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

    public static byte[] pprRoute(int ssid, PPRQuery query) {
        FlatBufferBuilder b = new FlatBufferBuilder();

        NamedLocation start = query.placeFrom;
        NamedLocation destination = query.placeTo;

        FootRoutingRequest.startDestinationsVector(b, 1);
        Position.createPosition(
                b, destination.lat, destination.lng);
        int destinationsOffset = b.endVector();

        int searchOptionsOffset = createPprSearchOptions(b, query.pprSearchOptions);

        FootRoutingRequest.startFootRoutingRequest(b);
        FootRoutingRequest.addStart(b,
                Position.createPosition(b, start.lat, start.lng));
        FootRoutingRequest.addDestinations(b, destinationsOffset);
        FootRoutingRequest.addSearchOptions(b, searchOptionsOffset);
        FootRoutingRequest.addSearchDirection(b, SearchDirection.Forward);
        FootRoutingRequest.addIncludeSteps(b, true);
        FootRoutingRequest.addIncludeEdges(b, false);
        FootRoutingRequest.addIncludePath(b, true);
        int footRoutingRequest = FootRoutingRequest.endFootRoutingRequest(b);

        b.finish(Message.createMessage(
                b, Destination.createDestination(
                        b, DestinationType.Module, b.createString("/ppr/route")),
                MsgContent.FootRoutingRequest, footRoutingRequest, ssid));

        return b.sizedByteArray();
    }

    public static byte[] intermodalRoute(int ssid, IntermodalQuery query,
                                         Date intervalBegin, Date intervalEnd,
                                         boolean extendIntervalEarlier,
                                         boolean extendIntervalLater,
                                         int minConnectionCount) {
        FlatBufferBuilder b = new FlatBufferBuilder();

        int start = createIntermodalPreTripStart(
                b, query.getPlaceFrom(),
                intervalBegin, intervalEnd,
                extendIntervalEarlier, extendIntervalLater,
                minConnectionCount);

        int destination = InputPosition.createInputPosition(
                b, query.getPlaceTo().lat, query.getPlaceTo().lng);

        int pprMode = createPPRMode(b, query);

        int intermodalRoutingRequest =
                IntermodalRoutingRequest.createIntermodalRoutingRequest(b,
                        IntermodalStart.IntermodalPretripStart, start,
                        IntermodalRoutingRequest.createStartModesVector(b, new int[]{pprMode}),
                        IntermodalDestination.InputPosition, destination,
                        IntermodalRoutingRequest.createDestinationModesVector(b, new int[]{pprMode}),
                        SearchType.Accessibility,
                        query.isArrival() ? SearchDir.Backward : SearchDir.Forward);

        b.finish(Message.createMessage(
                b, Destination.createDestination(
                        b, DestinationType.Module, b.createString("/intermodal")),
                MsgContent.IntermodalRoutingRequest, intermodalRoutingRequest, ssid));

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

    static private int createIntermodalPreTripStart(
            FlatBufferBuilder b,
            NamedLocation location,
            Date intervalBegin, Date intervalEnd,
            boolean extendIntervalEarlier, boolean extendIntervalLater,
            int minConnectionCount) {
        IntermodalPretripStart.startIntermodalPretripStart(b);
        IntermodalPretripStart.addPosition(b,
                Position.createPosition(b, location.lat, location.lng));
        IntermodalPretripStart.addInterval(
                b, Interval.createInterval(
                        b, intervalBegin.getTime() / 1000,
                        intervalEnd.getTime() / 1000));
        IntermodalPretripStart.addExtendIntervalEarlier(b, extendIntervalEarlier);
        IntermodalPretripStart.addExtendIntervalLater(b, extendIntervalLater);
        IntermodalPretripStart.addMinConnectionCount(b, minConnectionCount);
        return IntermodalPretripStart.endIntermodalPretripStart(b);
    }

    static private int createPPRMode(
            FlatBufferBuilder b, IntermodalQuery query) {
        return ModeWrapper.createModeWrapper(b, Mode.FootPPR,
                FootPPR.createFootPPR(b,
                        createPprSearchOptions(b, query.getPPRSettings().pprSearchOptions)));
    }

    static private int createPprSearchOptions(FlatBufferBuilder b, PprSearchOptions opt) {
        return SearchOptions.createSearchOptions(b,
                b.createString(opt.profile.getId()),
                opt.maxDuration * 60.0);
    }

    public static Message decode(byte[] buf) {
        return Message.getRootAsMessage(ByteBuffer.wrap(buf));
    }
}
