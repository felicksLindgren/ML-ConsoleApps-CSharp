using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CNTK;
using CNTKUtil;
using HeartDisease.Models;
using Microsoft.ML;
using XPlot.Plotly;

namespace HeartDisease
{
    class Program
    {
        static void Main(string[] args)
        {
            int[,] matrix = new int[,] { 
                {1,0,0,0,0},
                {0,0,0,0,0},
                {1,0,1,0,1},
                {1,0,1,0,1},
                {0,0,1,0,0}
            };

            bool[,] visited = new bool[matrix.GetLength(0), matrix.GetLength(1)];
            var sizes = new List<int>();

            for(var i = 0; i < matrix.GetLength(0); i++) {
                for(var j = 0; j < matrix.GetLength(1); j++) {
                    if(visited[i, j]) continue;

                    TraverseNode(i, j, matrix, visited, sizes);
                }
            }

            foreach(var i in sizes) {
                Console.WriteLine(i);
            }
        }

        public static void TraverseNode(int i, int j, int[,] matrix, bool[,] visited, List<int> sizes) {
            var currentRiverSize = 0;
            var nodesToExplore = new List<int[]>();

            nodesToExplore.Add(new int[] {i, j});

            while(nodesToExplore.Count != 0) {
                var currentNode = nodesToExplore[nodesToExplore.Count - 1];
                nodesToExplore.RemoveAt(nodesToExplore.Count - 1);

                i = currentNode[0];
                j = currentNode[1];

                if(visited[i,j]) continue;

                visited[i, j] = true;
                if(matrix[i,j] == 0) continue;

                currentRiverSize++;
                nodesToExplore.AddRange(GetUnvisitedNeighbours(i, j, matrix, visited));
            }

            if(currentRiverSize > 0) sizes.Add(currentRiverSize);
        }

        public static List<int[]> GetUnvisitedNeighbours(int i, int j, int[,] matrix, bool[,] visited) {
            var unvisitedNeighbours = new List<int[]>();

            if(i > 0 && !visited[i - 1, j]) unvisitedNeighbours.Add(new int[] {i - 1, j});
            if(i < matrix.GetLength(0) - 1 && !visited[i + 1, j]) unvisitedNeighbours.Add(new int[] {i + 1, j});
            if(j > 0 && !visited[i, j - 1]) unvisitedNeighbours.Add(new int[] { i, j - 1});
            if(j < matrix.GetLength(1) - 1 && !visited[i, j + 1]) unvisitedNeighbours.Add(new int[] { i, j + 1});

            return unvisitedNeighbours;
        }
    }
}
