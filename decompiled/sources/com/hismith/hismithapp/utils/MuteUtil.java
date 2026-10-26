package com.hismith.hismithapp.utils;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/* loaded from: classes3.dex */
public class MuteUtil {
    /* JADX WARN: Removed duplicated region for block: B:39:0x0071 A[Catch: IOException -> 0x006d, TRY_LEAVE, TryCatch #5 {IOException -> 0x006d, blocks: (B:46:0x0069, B:39:0x0071), top: B:45:0x0069 }] */
    /* JADX WARN: Removed duplicated region for block: B:45:0x0069 A[EXC_TOP_SPLITTER, SYNTHETIC] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public static void muteFile(String str, String str2) {
        FileOutputStream fileOutputStream;
        FileInputStream fileInputStream = null;
        try {
            try {
                FileInputStream fileInputStream2 = new FileInputStream(str);
                try {
                    fileOutputStream = new FileOutputStream(str2);
                    try {
                        byte[] bArr = new byte[640];
                        while (true) {
                            int read = fileInputStream2.read(bArr);
                            if (read == -1) {
                                fileInputStream2.close();
                                fileOutputStream.close();
                                return;
                            }
                            LogUtil.w(Arrays.toString(bArr));
                            if (AudioUtil.isMute(bArr, 16)) {
                                bArr = new byte[1024];
                                LogUtil.e("NULL");
                            } else {
                                LogUtil.e(Arrays.toString(bArr));
                                fileOutputStream.write(bArr, 0, read);
                            }
                        }
                    } catch (Exception e) {
                        e = e;
                        fileInputStream = fileInputStream2;
                        try {
                            e.printStackTrace();
                            if (fileInputStream != null) {
                                fileInputStream.close();
                            }
                            if (fileOutputStream != null) {
                                fileOutputStream.close();
                            }
                        } catch (Throwable th) {
                            th = th;
                            if (fileInputStream != null) {
                                try {
                                    fileInputStream.close();
                                } catch (IOException e2) {
                                    e2.printStackTrace();
                                    throw th;
                                }
                            }
                            if (fileOutputStream != null) {
                                fileOutputStream.close();
                            }
                            throw th;
                        }
                    } catch (Throwable th2) {
                        th = th2;
                        fileInputStream = fileInputStream2;
                        if (fileInputStream != null) {
                        }
                        if (fileOutputStream != null) {
                        }
                        throw th;
                    }
                } catch (Exception e3) {
                    e = e3;
                    fileOutputStream = null;
                } catch (Throwable th3) {
                    th = th3;
                    fileOutputStream = null;
                }
            } catch (IOException e4) {
                e4.printStackTrace();
            }
        } catch (Exception e5) {
            e = e5;
            fileOutputStream = null;
        } catch (Throwable th4) {
            th = th4;
            fileOutputStream = null;
        }
    }

    /* JADX WARN: Unsupported multi-entry loop pattern (BACK_EDGE: B:45:0x0062 -> B:20:0x0065). Please report as a decompilation issue!!! */
    public static List<byte[]> readFile(String str, String str2) {
        int available;
        ArrayList arrayList = new ArrayList();
        File file = new File(str);
        if (!file.exists()) {
            file.mkdirs();
        }
        File file2 = new File(file, str2);
        if (!file2.exists()) {
            try {
                file2.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        FileInputStream fileInputStream = null;
        FileInputStream fileInputStream2 = null;
        fileInputStream = null;
        try {
            try {
                try {
                    FileInputStream fileInputStream3 = new FileInputStream(file2);
                    int i = 24;
                    while (true) {
                        try {
                            available = fileInputStream3.available();
                            if (available <= 0) {
                                break;
                            }
                            byte[] bArr = new byte[i + 1600];
                            fileInputStream3.read(bArr);
                            if (i > 0) {
                                byte[] bArr2 = new byte[1600];
                                System.arraycopy(bArr, i, bArr2, 0, 1600);
                                bArr = bArr2;
                            }
                            arrayList.add(bArr);
                            i = 0;
                        } catch (Exception e2) {
                            e = e2;
                            fileInputStream2 = fileInputStream3;
                            e.printStackTrace();
                            fileInputStream = fileInputStream2;
                            if (fileInputStream2 != null) {
                                fileInputStream2.close();
                                fileInputStream = fileInputStream2;
                            }
                            return arrayList;
                        } catch (Throwable th) {
                            th = th;
                            fileInputStream = fileInputStream3;
                            if (fileInputStream != null) {
                                try {
                                    fileInputStream.close();
                                } catch (IOException e3) {
                                    e3.printStackTrace();
                                }
                            }
                            throw th;
                        }
                    }
                    fileInputStream3.close();
                    fileInputStream = available;
                } catch (Throwable th2) {
                    th = th2;
                }
            } catch (Exception e4) {
                e = e4;
            }
        } catch (IOException e5) {
            e5.printStackTrace();
            fileInputStream = fileInputStream;
        }
        return arrayList;
    }

    public static void writeFile(String str, String str2, byte[] bArr) {
        FileOutputStream fileOutputStream;
        File file = new File(str);
        if (!file.exists()) {
            file.mkdirs();
        }
        File file2 = new File(file, str2);
        if (!file2.exists()) {
            try {
                file2.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        FileOutputStream fileOutputStream2 = null;
        try {
            try {
                try {
                    fileOutputStream = new FileOutputStream(file2, true);
                } catch (Exception e2) {
                    e = e2;
                }
            } catch (Throwable th) {
                th = th;
            }
            try {
                fileOutputStream.write(bArr);
                LogUtil.e("写入数据长度：" + bArr.length);
                fileOutputStream.close();
            } catch (Exception e3) {
                e = e3;
                fileOutputStream2 = fileOutputStream;
                e.printStackTrace();
                if (fileOutputStream2 != null) {
                    fileOutputStream2.close();
                }
            } catch (Throwable th2) {
                th = th2;
                fileOutputStream2 = fileOutputStream;
                if (fileOutputStream2 != null) {
                    try {
                        fileOutputStream2.close();
                    } catch (IOException e4) {
                        e4.printStackTrace();
                    }
                }
                throw th;
            }
        } catch (IOException e5) {
            e5.printStackTrace();
        }
    }

    public static void writeString(String str, String str2, String str3) {
        FileOutputStream fileOutputStream;
        File file = new File(str);
        if (!file.exists()) {
            file.mkdirs();
        }
        File file2 = new File(file, str2);
        if (!file2.exists()) {
            try {
                file2.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        FileOutputStream fileOutputStream2 = null;
        try {
            try {
                try {
                    fileOutputStream = new FileOutputStream(file2, true);
                } catch (IOException e2) {
                    e2.printStackTrace();
                    return;
                }
            } catch (Exception e3) {
                e = e3;
            }
        } catch (Throwable th) {
            th = th;
        }
        try {
            LogUtil.e("写入：" + str2 + ", " + str3.length());
            fileOutputStream.write(str3.getBytes());
            fileOutputStream.close();
        } catch (Exception e4) {
            e = e4;
            fileOutputStream2 = fileOutputStream;
            e.printStackTrace();
            if (fileOutputStream2 != null) {
                fileOutputStream2.close();
            }
        } catch (Throwable th2) {
            th = th2;
            fileOutputStream2 = fileOutputStream;
            if (fileOutputStream2 != null) {
                try {
                    fileOutputStream2.close();
                } catch (IOException e5) {
                    e5.printStackTrace();
                }
            }
            throw th;
        }
    }
}
