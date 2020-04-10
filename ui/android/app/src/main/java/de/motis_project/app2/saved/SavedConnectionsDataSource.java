package de.motis_project.app2.saved;

import android.content.Context;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.database.sqlite.SQLiteStatement;

import com.google.flatbuffers.FlatBufferBuilder;
import com.squareup.sqlbrite.BriteDatabase;
import com.squareup.sqlbrite.SqlBrite;

import java.nio.ByteBuffer;
import java.util.List;

import motis.Connection;
import rx.Observable;
import rx.schedulers.Schedulers;

public class SavedConnectionsDataSource {
    private static class Table extends SQLiteOpenHelper {
        static final int DATABASE_VERSION = 1;
        static final String DATABASE_NAME = "connections.db";
        static final String TABLE = "connections";

        static final String COL_ID = "_id";
        static final String COL_DATA = "data";
        private static final String CREATE_LIST = ""
                + "CREATE TABLE " + TABLE + "("
                + COL_ID + " INTEGER PRIMARY KEY,"
                + COL_DATA + " TEXT NOT NULL"
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

    public class SavedConnection {
        final long id;
        final Connection con;

        public SavedConnection(long id, Connection con) {
            this.id = id;
            this.con = con;
        }
    }

    private final Table dbHelper;
    private final SqlBrite sqlBrite;
    private final BriteDatabase db;

    public SavedConnectionsDataSource(Context ctx) {
        sqlBrite = new SqlBrite.Builder()
                .logger(message -> System.out.println("DATABASE message = [" + message + "]"))
                .build();
        dbHelper = new Table(ctx);
        db = sqlBrite.wrapDatabaseHelper(dbHelper, Schedulers.io());
        db.setLoggingEnabled(true);
    }

    public void delete(long id) {
        SQLiteStatement s = dbHelper.getReadableDatabase().compileStatement(
                "DELETE FROM " + Table.TABLE
                        + " WHERE " + Table.COL_ID + " = ?");
        s.bindLong(1, id);
        db.executeUpdateDelete(Table.TABLE, s);
    }

    public void add(FlatBufferBuilder fbb) {
        SQLiteStatement s = dbHelper.getReadableDatabase().compileStatement(
                "INSERT INTO " + Table.TABLE
                        + " (" + Table.COL_DATA + ") VALUES (?)");
        s.bindBlob(1, fbb.dataBuffer().compact().array());
        db.executeInsert(Table.TABLE, s);
    }

    public Observable<List<SavedConnection>> getSavedConnections() {
        return db
                .createQuery(Table.TABLE, "SELECT * FROM " + Table.TABLE)
                .mapToList(c -> {
                    long id = c.getLong(c.getColumnIndex(Table.COL_ID));
                    ByteBuffer conBuf = ByteBuffer.wrap(c.getBlob(c.getColumnIndex(Table.COL_DATA)));
                    return new SavedConnection(id, Connection.getRootAsConnection(conBuf));
                });
    }
}
