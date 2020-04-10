package de.motis_project.app2.ppr;

import android.os.Parcel;
import android.os.Parcelable;

import com.google.android.libraries.places.compat.Place;
import com.google.android.gms.maps.model.LatLng;

import java.util.List;

public class NamedLocation implements Parcelable {
    public final String name;
    public final double lat;
    public final double lng;

    public NamedLocation(String name, double lat, double lng) {
        this.name = name;
        this.lat = lat;
        this.lng = lng;
    }

    public NamedLocation(Place place) {
        this(getPlaceName(place), place.getLatLng().latitude, place.getLatLng().longitude);
        System.out.println("Place: " + place);
        System.out.println("Place Name: " + place.getName());
        System.out.println("Place Address: " + place.getAddress());
        System.out.println("Place Id: " + place.getId());
        for (int placeType : place.getPlaceTypes()) {
            System.out.println("Place Type: " + placeType);
        }
    }

    public LatLng toLatLng() {
        return new LatLng(lat, lng);
    }

    @Override
    public String toString() {
        return "NamedLocation{" +
                "name='" + name + '\'' +
                ", lat=" + lat +
                ", lng=" + lng +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        NamedLocation that = (NamedLocation) o;

        if (Double.compare(that.lat, lat) != 0) return false;
        if (Double.compare(that.lng, lng) != 0) return false;
        return name != null ? name.equals(that.name) : that.name == null;
    }

    public boolean equalsLocation(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        NamedLocation that = (NamedLocation) o;

        if (Double.compare(that.lat, lat) != 0) return false;
        return Double.compare(that.lng, lng) == 0;
    }

    @Override
    public int hashCode() {
        int result;
        long temp;
        result = name != null ? name.hashCode() : 0;
        temp = Double.doubleToLongBits(lat);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        temp = Double.doubleToLongBits(lng);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        return result;
    }


    @Override
    public int describeContents() {
        return 0;
    }

    @Override
    public void writeToParcel(Parcel dest, int flags) {
        dest.writeString(this.name);
        dest.writeDouble(this.lat);
        dest.writeDouble(this.lng);
    }

    protected NamedLocation(Parcel in) {
        this.name = in.readString();
        this.lat = in.readDouble();
        this.lng = in.readDouble();
    }

    public static final Parcelable.Creator<NamedLocation> CREATOR = new Parcelable.Creator<NamedLocation>() {
        @Override
        public NamedLocation createFromParcel(Parcel source) {
            return new NamedLocation(source);
        }

        @Override
        public NamedLocation[] newArray(int size) {
            return new NamedLocation[size];
        }
    };

    private static String getPlaceName(Place place) {
        List<Integer> types = place.getPlaceTypes();
        if (types.size() == 1) {
            int type = types.get(0);
            if (type == Place.TYPE_OTHER
                    || type == Place.TYPE_ROUTE
                    || type == Place.TYPE_STREET_ADDRESS) {
                return place.getAddress().toString();
            }
        }
        return place.getName().toString();
    }
}
