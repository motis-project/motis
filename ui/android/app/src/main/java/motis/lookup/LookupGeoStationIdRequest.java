// automatically generated by the FlatBuffers compiler, do not modify

package motis.lookup;

import java.nio.*;
import java.lang.*;
import java.util.*;
import com.google.flatbuffers.*;

@SuppressWarnings("unused")
public final class LookupGeoStationIdRequest extends Table {
  public static LookupGeoStationIdRequest getRootAsLookupGeoStationIdRequest(ByteBuffer _bb) { return getRootAsLookupGeoStationIdRequest(_bb, new LookupGeoStationIdRequest()); }
  public static LookupGeoStationIdRequest getRootAsLookupGeoStationIdRequest(ByteBuffer _bb, LookupGeoStationIdRequest obj) { _bb.order(ByteOrder.LITTLE_ENDIAN); return (obj.__init(_bb.getInt(_bb.position()) + _bb.position(), _bb)); }
  public LookupGeoStationIdRequest __init(int _i, ByteBuffer _bb) { bb_pos = _i; bb = _bb; return this; }

  public String stationId() { int o = __offset(4); return o != 0 ? __string(o + bb_pos) : null; }
  public ByteBuffer stationIdAsByteBuffer() { return __vector_as_bytebuffer(4, 1); }
  public double minRadius() { int o = __offset(6); return o != 0 ? bb.getDouble(o + bb_pos) : 0.0; }
  public double maxRadius() { int o = __offset(8); return o != 0 ? bb.getDouble(o + bb_pos) : 0.0; }

  public static int createLookupGeoStationIdRequest(FlatBufferBuilder builder,
      int station_idOffset,
      double min_radius,
      double max_radius) {
    builder.startObject(3);
    LookupGeoStationIdRequest.addMaxRadius(builder, max_radius);
    LookupGeoStationIdRequest.addMinRadius(builder, min_radius);
    LookupGeoStationIdRequest.addStationId(builder, station_idOffset);
    return LookupGeoStationIdRequest.endLookupGeoStationIdRequest(builder);
  }

  public static void startLookupGeoStationIdRequest(FlatBufferBuilder builder) { builder.startObject(3); }
  public static void addStationId(FlatBufferBuilder builder, int stationIdOffset) { builder.addOffset(0, stationIdOffset, 0); }
  public static void addMinRadius(FlatBufferBuilder builder, double minRadius) { builder.addDouble(1, minRadius, 0.0); }
  public static void addMaxRadius(FlatBufferBuilder builder, double maxRadius) { builder.addDouble(2, maxRadius, 0.0); }
  public static int endLookupGeoStationIdRequest(FlatBufferBuilder builder) {
    int o = builder.endObject();
    return o;
  }
};
