using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Drawing;

namespace ggg2png
{
   class Program
   {
      static void Main(string[] args)
      {
         int shift = 0;
         bool color = false;
         if (args[shift] == "c")
         {
            color = true;
            shift++;
         }
         string[] files = Directory.GetFiles(args[shift++]);
         foreach (var file in files)
         {
            var path = Path.GetFullPath(Path.Combine(Path.GetDirectoryName(file), "..\\pngs"));
            var dstname = Path.Combine(path, Path.GetFileNameWithoutExtension(file) + ".png");
            //var dstname = file.Replace(".raw", ".png");
            FileStream fs = new FileStream(file, FileMode.Open);
            BinaryReader br = new BinaryReader(fs);
            int width = br.ReadInt32();
            int height = br.ReadInt32();
            int xdpi = br.ReadInt32();
            int ydpi = br.ReadInt32();
            Bitmap bmp = new Bitmap(width, height);
            for (int y = 0; y < height; ++y)
            {
               for (int x = 0; x < width; ++x)
               {
                  if (color)
                  {
                     int b = (int)(br.ReadUInt32() & 0xFFFFFFu);
                     bmp.SetPixel(x, y, Color.FromArgb((b >> 16) & 0xFF, (b >> 8) & 0xFF, b & 0xFF));
                  }
                  else
                  {
                     int b = (int)(br.ReadUInt32() & 0xFFu);
                     bmp.SetPixel(x, y, Color.FromArgb(b, b, b));
                  }
               }
            }
            bmp.Save(dstname, System.Drawing.Imaging.ImageFormat.Png);
         }
      }
   }
}
