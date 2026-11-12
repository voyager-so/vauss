// Parse the exported OBJ numerically so Unity's model-import axis conversion cannot
// disagree with UnityGaussianSplatting's PLY reader (which preserves XYZ).
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

public static class SplatCollisionSetup
{
    [MenuItem("Tools/Splat/Create Static Collision")]
    public static void Create()
    {
        const string source = "Assets/SplatScene/collision.obj";
        const string asset = "Assets/SplatScene/CollisionMesh.asset";
        if (!File.Exists(source)) { Debug.LogError("Import " + source + " first."); return; }
        var vertices = new List<Vector3>();
        var triangles = new List<int>();
        foreach (var line in File.ReadLines(source))
        {
            var fields = line.Split(new[] { ' ' }, System.StringSplitOptions.RemoveEmptyEntries);
            if (fields.Length != 4) continue;
            if (fields[0] == "v") vertices.Add(new Vector3(
                float.Parse(fields[1], CultureInfo.InvariantCulture),
                float.Parse(fields[2], CultureInfo.InvariantCulture),
                float.Parse(fields[3], CultureInfo.InvariantCulture)));
            if (fields[0] == "f") for (int i = 1; i < 4; ++i)
                triangles.Add(int.Parse(fields[i], CultureInfo.InvariantCulture) - 1);
        }
        if (triangles.Count == 0) { Debug.LogWarning("No opaque collision surface in this export."); return; }
        var mesh = AssetDatabase.LoadAssetAtPath<Mesh>(asset);
        bool createAsset = mesh == null;
        if (createAsset) mesh = new Mesh();
        else { Undo.RecordObject(mesh, "Update Splat Collision"); mesh.Clear(); }
        mesh.name = "Splat Collision";
        mesh.indexFormat = IndexFormat.UInt32;
        mesh.SetVertices(vertices);
        mesh.SetTriangles(triangles, 0);
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        if (createAsset) AssetDatabase.CreateAsset(mesh, asset);
        else EditorUtility.SetDirty(mesh);
        AssetDatabase.SaveAssets();
        var root = new GameObject("Splat Collision");
        Undo.RegisterCreatedObjectUndo(root, "Create Splat Collision");
        root.isStatic = true;
        root.AddComponent<MeshFilter>().sharedMesh = mesh;
        var renderer = root.AddComponent<MeshRenderer>();
        renderer.sharedMaterial = AssetDatabase.GetBuiltinExtraResource<Material>("Default-Material.mat");
        renderer.enabled = false;
        var collider = root.AddComponent<MeshCollider>();
        collider.sharedMesh = mesh;
        collider.convex = false;
        Selection.activeGameObject = root;
    }
}
