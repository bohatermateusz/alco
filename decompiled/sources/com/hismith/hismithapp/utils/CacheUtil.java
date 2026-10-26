package com.hismith.hismithapp.utils;

import android.content.Context;
import android.graphics.Bitmap;
import com.bumptech.glide.load.Key;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.channels.FileChannel;
import java.util.Objects;

/* loaded from: classes3.dex */
public class CacheUtil {
    private static File CacheRoot;

    public static boolean adsExists(Context context) {
        File file = new File(context.getCacheDir().getPath() + "/hismith_ads/");
        if (file.exists()) {
            File[] listFiles = file.listFiles();
            if (listFiles != null && listFiles.length > 0) {
                File file2 = new File(listFiles[0], "HISMITH_ADS_IMAGE.jpg");
                File file3 = new File(listFiles[0], "HISMITH_ADS_CONTENT.html");
                if (file2.exists() && file3.exists()) {
                    return true;
                }
            }
            deleteAds(context);
        }
        return false;
    }

    public static boolean adsVersionExists(Context context, String str) {
        new File(context.getCacheDir().getPath() + "/hismith_ads/");
        File file = new File(context.getCacheDir().getPath() + "/hismith_ads/" + str);
        if (file.exists()) {
            return new File(file, "HISMITH_ADS_IMAGE.jpg").exists() && new File(file, "HISMITH_ADS_CONTENT.html").exists();
        }
        return false;
    }

    public static void deleteAds(Context context) {
        LogUtil.e("删除本地所有广告页");
        try {
            deleteFilesByDirectory(new File(context.getCacheDir().getPath() + "/hismith_ads/"));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void deleteAdsVersionOther(Context context, String str) {
        LogUtil.e("删除广告页版本号：" + str + "以外的广告");
        try {
            File file = new File(context.getCacheDir().getPath() + "/hismith_ads/");
            if (file.exists() && file.isDirectory()) {
                for (File file2 : (File[]) Objects.requireNonNull(file.listFiles())) {
                    if (!file2.getName().equals(str)) {
                        deleteFilesByDirectory(file2);
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void deleteCrash(Context context, String str) {
        try {
            File file = new File(context.getCacheDir().getPath() + "/crash_collection/", str);
            if (file.exists() && file.isFile()) {
                file.delete();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void deleteFilesByDirectory(File file) {
        if (file != null) {
            try {
                if (file.exists() && file.isDirectory()) {
                    for (File file2 : file.listFiles()) {
                        if (file2.isDirectory()) {
                            deleteFilesByDirectory(file2);
                        } else {
                            file2.delete();
                        }
                    }
                    file.delete();
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    public static void deleteJson(Context context, String str) {
        try {
            CacheRoot = context.getFilesDir();
            File file = new File(CacheRoot, str);
            if (file.exists() && file.isFile()) {
                file.delete();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static long getPlaySVGModifyTime(Context context, String str) {
        File file = new File(context.getFilesDir() + "/playSVG");
        if (!file.exists()) {
            file.mkdir();
        }
        File file2 = new File(file, str + ".svg");
        if (file2.exists()) {
            return file2.lastModified();
        }
        return 0L;
    }

    public static boolean playSVGExists(Context context, String str) {
        File file = new File(context.getFilesDir() + "/playSVG");
        if (!file.exists()) {
            file.mkdir();
        }
        File file2 = new File(file, str + ".svg");
        return file2.exists() && file2.length() > 0;
    }

    public static String readAds(Context context) {
        File[] listFiles;
        File file = new File(context.getCacheDir().getPath() + "/hismith_ads/");
        return (!file.exists() || (listFiles = file.listFiles()) == null || listFiles.length <= 0) ? "" : readFile(new File(listFiles[0], "HISMITH_ADS_CONTENT.html"));
    }

    /* JADX WARN: Code restructure failed: missing block: B:20:0x0071, code lost:
    
        if (r5 == null) goto L37;
     */
    /* JADX WARN: Code restructure failed: missing block: B:21:0x0066, code lost:
    
        r5.close();
     */
    /* JADX WARN: Code restructure failed: missing block: B:29:0x0064, code lost:
    
        if (r5 == null) goto L37;
     */
    /* JADX WARN: Multi-variable type inference failed */
    /* JADX WARN: Type inference failed for: r5v4 */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static String readCrash(Context context, String str) {
        FileReader fileReader;
        IOException e;
        FileNotFoundException e2;
        String str2 = context.getCacheDir().getPath() + "/crash_collection/";
        StringBuilder sb = new StringBuilder();
        File file = new File(str2, str);
        FileReader fileReader2 = null;
        try {
        } catch (Throwable th) {
            th = th;
        }
        try {
            try {
                if (file.exists() && file.isFile()) {
                    fileReader = new FileReader(file);
                    try {
                        char[] cArr = new char[1024];
                        while (true) {
                            int read = fileReader.read(cArr);
                            if (read == -1) {
                                break;
                            }
                            sb.append(cArr, 0, read);
                        }
                        fileReader2 = fileReader;
                    } catch (FileNotFoundException e3) {
                        e2 = e3;
                        e2.printStackTrace();
                    } catch (IOException e4) {
                        e = e4;
                        e.printStackTrace();
                    }
                }
            } catch (IOException e5) {
                e5.printStackTrace();
            }
        } catch (FileNotFoundException e6) {
            fileReader = null;
            e2 = e6;
        } catch (IOException e7) {
            fileReader = null;
            e = e7;
        } catch (Throwable th2) {
            str = 0;
            th = th2;
            if (str != 0) {
                try {
                    str.close();
                } catch (IOException e8) {
                    e8.printStackTrace();
                }
            }
            throw th;
        }
        if (fileReader2 != null) {
            fileReader2.close();
        }
        return sb.toString();
    }

    /* JADX WARN: Code restructure failed: missing block: B:11:0x0055, code lost:
    
        r4 = move-exception;
     */
    /* JADX WARN: Code restructure failed: missing block: B:12:0x0056, code lost:
    
        r4.printStackTrace();
     */
    /* JADX WARN: Code restructure failed: missing block: B:31:0x0044, code lost:
    
        if (r1 == null) goto L43;
     */
    /* JADX WARN: Code restructure failed: missing block: B:35:0x003a, code lost:
    
        if (r1 == null) goto L43;
     */
    /* JADX WARN: Code restructure failed: missing block: B:4:0x0052, code lost:
    
        if (r1 != null) goto L45;
     */
    /* JADX WARN: Code restructure failed: missing block: B:9:0x003c, code lost:
    
        r1.close();
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    private static String readFile(File file) {
        StringBuilder sb = new StringBuilder();
        FileReader fileReader = null;
        try {
            if (file != null) {
                try {
                    if (file.exists() && file.isFile()) {
                        FileReader fileReader2 = new FileReader(file);
                        try {
                            char[] cArr = new char[1024];
                            while (true) {
                                int read = fileReader2.read(cArr);
                                if (read == -1) {
                                    break;
                                }
                                sb.append(cArr, 0, read);
                            }
                            fileReader = fileReader2;
                        } catch (FileNotFoundException e) {
                            e = e;
                            fileReader = fileReader2;
                            e.printStackTrace();
                        } catch (IOException e2) {
                            e = e2;
                            fileReader = fileReader2;
                            e.printStackTrace();
                        } catch (Throwable th) {
                            th = th;
                            fileReader = fileReader2;
                            if (fileReader != null) {
                                try {
                                    fileReader.close();
                                } catch (IOException e3) {
                                    e3.printStackTrace();
                                }
                            }
                            throw th;
                        }
                    }
                } catch (FileNotFoundException e4) {
                    e = e4;
                } catch (IOException e5) {
                    e = e5;
                }
            }
            return sb.toString();
        } catch (Throwable th2) {
            th = th2;
        }
    }

    /* JADX WARN: Code restructure failed: missing block: B:20:0x0059, code lost:
    
        if (r1 == null) goto L37;
     */
    /* JADX WARN: Code restructure failed: missing block: B:21:0x004f, code lost:
    
        r1.close();
     */
    /* JADX WARN: Code restructure failed: missing block: B:29:0x004d, code lost:
    
        if (r1 == null) goto L37;
     */
    /* JADX WARN: Multi-variable type inference failed */
    /* JADX WARN: Removed duplicated region for block: B:37:0x0065 A[EXC_TOP_SPLITTER, SYNTHETIC] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static String readJson(Context context, String str) {
        FileReader fileReader;
        IOException e;
        FileNotFoundException e2;
        CacheRoot = context.getFilesDir();
        StringBuilder sb = new StringBuilder();
        File file = CacheRoot;
        File file2 = new File(file, str);
        FileReader fileReader2 = 0;
        r4 = null;
        FileReader fileReader3 = null;
        try {
            try {
                try {
                    if (file2.exists() && file2.isFile()) {
                        fileReader = new FileReader(file2);
                        try {
                            char[] cArr = new char[1024];
                            while (true) {
                                int read = fileReader.read(cArr);
                                if (read == -1) {
                                    break;
                                }
                                sb.append(cArr, 0, read);
                            }
                            fileReader3 = fileReader;
                        } catch (FileNotFoundException e3) {
                            e2 = e3;
                            e2.printStackTrace();
                        } catch (IOException e4) {
                            e = e4;
                            e.printStackTrace();
                        }
                    }
                } catch (Throwable th) {
                    th = th;
                    fileReader2 = file;
                    if (fileReader2 != 0) {
                        try {
                            fileReader2.close();
                        } catch (IOException e5) {
                            e5.printStackTrace();
                        }
                    }
                    throw th;
                }
            } catch (FileNotFoundException e6) {
                fileReader = null;
                e2 = e6;
            } catch (IOException e7) {
                fileReader = null;
                e = e7;
            } catch (Throwable th2) {
                th = th2;
                if (fileReader2 != 0) {
                }
                throw th;
            }
            if (fileReader3 != null) {
                fileReader3.close();
            }
        } catch (IOException e8) {
            e8.printStackTrace();
        }
        return sb.toString();
    }

    public static void saveAds(Context context, String str, Bitmap bitmap, String str2) {
        File file = new File(context.getCacheDir().getPath() + "/hismith_ads/", str);
        if (file.exists()) {
            return;
        }
        if (!file.mkdirs()) {
            LogUtil.e("创建广告页文件夹失败");
            return;
        }
        LogUtil.e("创建广告页文件夹成功！");
        try {
            BufferedOutputStream bufferedOutputStream = new BufferedOutputStream(new FileOutputStream(new File(file.getPath(), "HISMITH_ADS_IMAGE.jpg")));
            bitmap.compress(Bitmap.CompressFormat.JPEG, 80, bufferedOutputStream);
            LogUtil.e("保存广告页图片成功！");
            bufferedOutputStream.flush();
            bufferedOutputStream.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        try {
            FileOutputStream fileOutputStream = new FileOutputStream(new File(file.getPath(), "HISMITH_ADS_CONTENT.html"));
            fileOutputStream.write(str2.getBytes(Key.STRING_CHARSET_NAME));
            LogUtil.e("保存广告页标签内容成功！");
            fileOutputStream.flush();
            fileOutputStream.close();
        } catch (Exception e2) {
            e2.printStackTrace();
            LogUtil.e("保存广告页异常");
        }
    }

    public static void writeCrash(Context context, Throwable th) {
        StringBuffer stringBuffer = new StringBuffer();
        StringWriter stringWriter = new StringWriter();
        PrintWriter printWriter = new PrintWriter(stringWriter);
        th.printStackTrace(printWriter);
        for (Throwable cause = th.getCause(); cause != null; cause = cause.getCause()) {
            cause.printStackTrace(printWriter);
        }
        printWriter.close();
        stringBuffer.append(stringWriter.toString());
        try {
            File file = new File(context.getCacheDir().getPath() + "/crash_collection/");
            if (!file.exists()) {
                file.mkdirs();
            }
            File file2 = new File(file, "SocketFailure.txt");
            if (!file2.exists()) {
                file2.createNewFile();
            }
            FileOutputStream fileOutputStream = new FileOutputStream(file2, true);
            fileOutputStream.write(stringBuffer.toString().getBytes(Key.STRING_CHARSET_NAME));
            fileOutputStream.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e2) {
            e2.printStackTrace();
        }
    }

    /* JADX WARN: Removed duplicated region for block: B:28:0x0059 A[EXC_TOP_SPLITTER, SYNTHETIC] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static void writeJson(Context context, String str, String str2, boolean z) {
        Throwable th;
        FileOutputStream fileOutputStream;
        IOException e;
        FileNotFoundException e2;
        CacheRoot = context.getFilesDir();
        FileOutputStream fileOutputStream2 = null;
        try {
            try {
                try {
                    File file = new File(CacheRoot, str2);
                    boolean exists = file.exists();
                    fileOutputStream = new FileOutputStream(file, z);
                    if (z && exists) {
                        try {
                            FileChannel channel = fileOutputStream.getChannel();
                            channel.truncate(channel.position() - 4);
                        } catch (FileNotFoundException e3) {
                            e2 = e3;
                            e2.printStackTrace();
                            if (fileOutputStream == null) {
                                return;
                            }
                            fileOutputStream.close();
                            return;
                        } catch (IOException e4) {
                            e = e4;
                            e.printStackTrace();
                            if (fileOutputStream == null) {
                                return;
                            }
                            fileOutputStream.close();
                            return;
                        }
                    }
                    fileOutputStream.write(str.getBytes());
                    fileOutputStream.close();
                } catch (Throwable th2) {
                    th = th2;
                    if (0 != 0) {
                        try {
                            fileOutputStream2.close();
                        } catch (IOException e5) {
                            e5.printStackTrace();
                        }
                    }
                    throw th;
                }
            } catch (FileNotFoundException e6) {
                fileOutputStream = null;
                e2 = e6;
            } catch (IOException e7) {
                fileOutputStream = null;
                e = e7;
            } catch (Throwable th3) {
                th = th3;
                if (0 != 0) {
                }
                throw th;
            }
        } catch (IOException e8) {
            e8.printStackTrace();
        }
    }
}
