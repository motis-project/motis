package de.motis_project.app2.query.guesser;

import android.content.ContentValues;
import android.content.Context;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;

import com.squareup.sqlbrite.BriteDatabase;
import com.squareup.sqlbrite.QueryObservable;
import com.squareup.sqlbrite.SqlBrite;

import java.util.List;

import rx.Observable;
import rx.schedulers.Schedulers;

public class FavoritesDataSource {
    private class Table extends SQLiteOpenHelper {
        static final int DATABASE_VERSION = 1;
        static final String DATABASE_NAME = "stations.db";

        static final String TABLE = "favorites";
        static final String COL_STATION_ID = "_id";
        static final String COL_STATION_NAME = "name";
        static final String COL_SELECTED_COUNT = "priority";

        private static final String CREATE_LIST = ""
            + "CREATE TABLE " + TABLE + "("
            + COL_STATION_ID + " TEXT NOT NULL PRIMARY KEY,"
            + COL_STATION_NAME + " TEXT NOT NULL,"
            + COL_SELECTED_COUNT + " INTEGER NOT NULL DEFAULT 0"
            + ")";

        Table(Context context) {
            super(context, DATABASE_NAME, null, DATABASE_VERSION);
        }

        @Override
        public void onCreate(SQLiteDatabase db) {
            db.execSQL(CREATE_LIST);
        }

        @Override
        public void onUpgrade(SQLiteDatabase sqLiteDatabase, int i, int i1) {
        }
    }

    private static final String SQL_GET_TOP = "" +
        "SELECT * FROM " + Table.TABLE +
        " WHERE " + Table.COL_STATION_NAME + " LIKE ?" +
        " ORDER BY " + Table.COL_SELECTED_COUNT +
        " DESC LIMIT 5";

    private final SqlBrite sqlBrite;
    private final BriteDatabase db;

    public FavoritesDataSource(Context ctx) {
        sqlBrite = new SqlBrite.Builder()
            .logger(message -> System.out.println("DATABASE message = [" + message + "]"))
            .build();
        db = sqlBrite.wrapDatabaseHelper(new Table(ctx), Schedulers.io());
        db.setLoggingEnabled(true);
    }

    public void addOrIncrement(String eva, String stationName) {
        try (BriteDatabase.Transaction t = db.newTransaction()) {
            ContentValues cv = new ContentValues();
            cv.put(Table.COL_STATION_ID, eva);
            cv.put(Table.COL_STATION_NAME, stationName);
            cv.put(Table.COL_SELECTED_COUNT, 0);
            db.insert(Table.TABLE, cv);
            cv.clear();
            db.execute(
                "UPDATE " + Table.TABLE +
                    " SET " + Table.COL_SELECTED_COUNT + " = " +
                    Table.COL_SELECTED_COUNT + " + 1 " +
                    " WHERE " + Table.COL_STATION_ID + " = ?", eva);
            t.markSuccessful();
        }
    }

    public Observable<List<StationGuess>> getFavorites(CharSequence queryString) {
        QueryObservable obs = db.createQuery(Table.TABLE, SQL_GET_TOP,
            "%" + queryString.toString().replace("%", "%%") + "%");
        return obs.mapToList(c -> {
            String eva = c.getString(c.getColumnIndex(Table.COL_STATION_ID));
            String name = c.getString(c.getColumnIndex(Table.COL_STATION_NAME));
            int count = c.getInt(c.getColumnIndex(Table.COL_SELECTED_COUNT));
            return new StationGuess(eva, name, count, StationGuess.FAVORITE_GUESS);
        });
    }
}
